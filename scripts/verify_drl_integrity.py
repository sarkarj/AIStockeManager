from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.drl.drl_engine import evaluate_drl  # noqa: E402
from app.core.replay.artifact_store import compute_policy_hash, load_artifact, save_artifact  # noqa: E402

try:
    from deepdiff import DeepDiff
except Exception:
    DeepDiff = None  # type: ignore[assignment]

POLICY_PATH = ROOT / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"
FIXTURES_PATH = ROOT / "app" / "data" / "fixtures" / "drl_fixtures.yaml"
NOW_ISO = "2026-02-11T12:00:00-05:00"


class IntegrityFailure(Exception):
    pass


def main() -> None:
    failures: list[dict[str, Any]] = []

    _record_check("CHECK 1: FIXTURE LOCK", _check_fixture_lock, failures)
    _record_check("CHECK 2: INVARIANTS LOCK", _check_invariants_lock, failures)
    _record_check("CHECK 3: POLICY ROUNDTRIP LOCK", _check_policy_roundtrip_lock, failures)
    _record_check("CHECK 4: SEMANTICS LOCK", _check_semantics_lock, failures)

    if failures:
        print("FAIL")
        for idx, failure in enumerate(failures, start=1):
            print(f"[{idx}] {failure['name']}")
            print(f"    reason: {failure['reason']}")
            details = failure.get("details")
            if details is not None:
                details_json = json.dumps(details, ensure_ascii=True, indent=2)
                for line in details_json.splitlines():
                    print(f"    {line}")
        raise SystemExit(1)

    print("PASS")
    raise SystemExit(0)


def _record_check(name: str, fn: Any, failures: list[dict[str, Any]]) -> None:
    try:
        info = fn()
        summary = info.get("summary", "ok") if isinstance(info, dict) else "ok"
        print(f"[PASS] {name} | {summary}")
    except IntegrityFailure as exc:
        payload = {"name": name, "reason": str(exc)}
        details = getattr(exc, "details", None)
        if details is not None:
            payload["details"] = details
        failures.append(payload)
    except Exception as exc:
        failures.append({"name": name, "reason": f"Unexpected error: {exc}"})


def _check_fixture_lock() -> dict:
    fixtures = _load_fixtures()
    mismatches: list[dict[str, Any]] = []

    for fx in fixtures:
        inputs = fx["inputs"]
        expected = fx["expected"]
        actual = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inputs, now_iso=NOW_ISO)

        missing_keys = [
            key
            for key in [
                "regime_1D",
                "regime_1W",
                "action_final",
                "confidence_cap",
                "gates_triggered",
                "conflicts",
                "decision_trace",
            ]
            if key not in actual
        ]
        if missing_keys:
            mismatches.append({"fixture": fx["id"], "missing_keys": missing_keys})
            continue

        if actual["regime_1D"] != expected["regime_1D"]:
            mismatches.append({"fixture": fx["id"], "field": "regime_1D", "expected": expected["regime_1D"], "actual": actual["regime_1D"]})
        if actual["regime_1W"] != expected["regime_1W"]:
            mismatches.append({"fixture": fx["id"], "field": "regime_1W", "expected": expected["regime_1W"], "actual": actual["regime_1W"]})
        if actual["action_final"] != expected["action_final"]:
            mismatches.append({"fixture": fx["id"], "field": "action_final", "expected": expected["action_final"], "actual": actual["action_final"]})

        if float(actual["confidence_cap"]) > float(expected["confidence_cap_max"]):
            mismatches.append(
                {
                    "fixture": fx["id"],
                    "field": "confidence_cap",
                    "expected_max": expected["confidence_cap_max"],
                    "actual": actual["confidence_cap"],
                }
            )

        expected_gates = set(expected.get("gates_triggered", []))
        actual_gates = set(actual.get("gates_triggered", []))
        if not expected_gates.issubset(actual_gates):
            mismatches.append(
                {
                    "fixture": fx["id"],
                    "field": "gates_triggered",
                    "expected_subset": sorted(expected_gates),
                    "actual": sorted(actual_gates),
                }
            )

        expected_conflicts = set(expected.get("conflicts_contains", []))
        actual_conflicts = set(actual.get("conflicts", []))
        if not expected_conflicts.issubset(actual_conflicts):
            mismatches.append(
                {
                    "fixture": fx["id"],
                    "field": "conflicts",
                    "expected_subset": sorted(expected_conflicts),
                    "actual": sorted(actual_conflicts),
                }
            )

        trace = actual["decision_trace"]
        required_trace = ["policy_id", "policy_version", "timestamp", "ticker", "score_final", "base_action", "action_final"]
        missing_trace = [k for k in required_trace if k not in trace]
        if missing_trace:
            mismatches.append({"fixture": fx["id"], "field": "decision_trace_missing", "missing": missing_trace})

    if mismatches:
        exc = IntegrityFailure("Fixture lock failed")
        exc.details = {"mismatches": mismatches}
        raise exc

    return {"summary": f"{len(fixtures)}/{len(fixtures)} fixtures matched"}


def _check_invariants_lock() -> dict:
    checks = _run_invariants()
    failed = [c for c in checks if not c["passed"]]
    if failed:
        exc = IntegrityFailure("Invariant lock failed")
        exc.details = {"failed": failed}
        raise exc
    return {"summary": f"{len(checks)}/{len(checks)} invariants passed"}


def _check_policy_roundtrip_lock() -> dict:
    direct_hash = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()[:16]
    fn_hash = compute_policy_hash(str(POLICY_PATH))

    if direct_hash != fn_hash:
        exc = IntegrityFailure("compute_policy_hash mismatch")
        exc.details = {"expected": direct_hash, "actual": fn_hash}
        raise exc

    fixture_inputs = _fixture_by_id("F10_ALL_ALIGNED_BULL_ACCUMULATE")["inputs"]
    drl_result = evaluate_drl(policy_path=str(POLICY_PATH), inputs=fixture_inputs, now_iso=NOW_ISO)
    context_pack = {
        "meta": {"ticker": fixture_inputs["ticker"], "generated_at": NOW_ISO, "interval": "1h", "lookback_days": 60, "data_quality": {"overall_stale": False, "notes": []}},
        "prices": {"as_of": fixture_inputs["as_of"], "bars": []},
        "indicators": {
            "as_of": fixture_inputs["as_of"],
            "metrics": {k: v for k, v in fixture_inputs.items() if k not in {"ticker", "as_of"}},
        },
        "drl": {"result": drl_result, "decision_trace": drl_result["decision_trace"]},
    }

    path = save_artifact(
        ticker=fixture_inputs["ticker"],
        policy_path=str(POLICY_PATH),
        context_pack=context_pack,
        now_iso=NOW_ISO,
        notes=["integrity-policy-hash-check"],
    )
    artifact = load_artifact(path)
    stored_hash = str(artifact.get("meta", {}).get("policy_hash", ""))

    if stored_hash != direct_hash:
        exc = IntegrityFailure("Artifact policy_hash mismatch")
        exc.details = {"expected": direct_hash, "actual": stored_hash, "path": path}
        raise exc

    return {"summary": f"policy hash verified ({direct_hash})"}


def _check_semantics_lock() -> dict:
    policy = _load_policy()
    cases = _semantic_cases()
    mismatches: list[dict[str, Any]] = []

    for case in cases:
        inputs = case["inputs"]
        expected = _reference_drl(policy=policy, inputs=inputs, now_iso=NOW_ISO)
        actual_full = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inputs, now_iso=NOW_ISO)
        actual = _actual_subset(actual_full)

        if expected != actual:
            mismatches.append(
                {
                    "case": case["name"],
                    "expected": expected,
                    "actual": actual,
                    "diff": _diff_dict(expected, actual),
                }
            )

    if mismatches:
        exc = IntegrityFailure("Semantics lock failed")
        exc.details = {"count": len(mismatches), "cases": mismatches[:5]}
        raise exc

    return {"summary": f"{len(cases)} synthetic cases matched reference DRL"}


def _semantic_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for value in [29.0, 30.0, 31.0]:
        inputs = _base_inputs(f"RSI_{int(value)}")
        inputs["rsi_14"] = value
        cases.append({"name": f"zone_rsi_{value}", "inputs": inputs})

    for value in [14.0, 15.0, 16.0]:
        inputs = _base_inputs(f"STOCH_{int(value)}")
        inputs["stoch_k"] = value
        cases.append({"name": f"zone_stoch_{value}", "inputs": inputs})

    for value in [24.0, 25.0, 26.0]:
        inputs = _base_inputs(f"ADX_{int(value)}")
        inputs["adx_14"] = value
        cases.append({"name": f"zone_adx_{value}", "inputs": inputs})

    for value in [5.9, 6.0, 6.1]:
        inputs = _base_inputs(f"ATR_{str(value).replace('.', '_')}")
        inputs["atr_pct"] = value
        cases.append({"name": f"zone_atr_{value}", "inputs": inputs})

    for value in [19.0, 20.0, 21.0]:
        inputs = _base_inputs(f"VROC_{int(value)}")
        inputs["vroc_14"] = value
        cases.append({"name": f"zone_vroc_{value}", "inputs": inputs})

    fx_map = {fx["id"]: fx for fx in _load_fixtures()}
    cases.append({"name": "gate_oversold_trigger", "inputs": dict(fx_map["F01_OVERSOLD_BEAR_ADX_WEAK_WAIT"]["inputs"])})
    cases.append({"name": "gate_oversold_exception", "inputs": dict(fx_map["F02_OVERSOLD_BEAR_ADX_STRONG_REDUCE_LOWCONF"]["inputs"])})
    cases.append({"name": "gate_overbought", "inputs": dict(fx_map["F03_OVERBOUGHT_BULL_ADX_STRONG_WAIT"]["inputs"])})
    cases.append({"name": "gate_timeframe_conflict", "inputs": dict(fx_map["F04_DAILY_BEAR_WEEKLY_BULL_WAIT"]["inputs"])})
    cases.append({"name": "gate_high_vol", "inputs": dict(fx_map["F06_ATR_EXTREME_CAP_CONF"]["inputs"])})
    cases.append({"name": "gate_low_participation", "inputs": dict(fx_map["F07_VROC_FALLING_LOW_PARTICIPATION"]["inputs"])})

    gov1 = _base_inputs("GOV1")
    gov1.update(
        {
            "price_last": 100.0,
            "ema_50": 110.0,
            "sma_200": 90.0,
            "rsi_14": 29.0,
            "macd": -2.0,
            "macd_signal": -1.0,
            "stoch_k": 10.0,
            "adx_14": 24.0,
            "vroc_14": -15.0,
            "atr_pct": 2.0,
            "supertrend_dir_1D": "BEAR",
            "supertrend_dir_1W": "BULL",
        }
    )
    cases.append({"name": "governor_weekly_bull_blocks_reduce", "inputs": gov1})

    gov2 = _base_inputs("GOV2")
    gov2.update(
        {
            "price_last": 100.0,
            "ema_50": 95.0,
            "sma_200": 110.0,
            "rsi_14": 60.0,
            "macd": 2.0,
            "macd_signal": 1.0,
            "stoch_k": 60.0,
            "adx_14": 30.0,
            "vroc_14": 25.0,
            "atr_pct": 2.0,
            "supertrend_dir_1D": "BULL",
            "supertrend_dir_1W": "BEAR",
        }
    )
    cases.append({"name": "governor_weekly_bear_blocks_accumulate", "inputs": gov2})

    return cases


def _base_inputs(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "as_of": NOW_ISO,
        "price_last": 320.0,
        "ema_50": 300.0,
        "sma_200": 250.0,
        "rsi_14": 55.0,
        "macd": 1.0,
        "macd_signal": 0.5,
        "stoch_k": 55.0,
        "adx_14": 24.0,
        "vroc_14": 10.0,
        "atr_pct": 3.0,
        "supertrend_dir_1D": "BULL",
        "supertrend_dir_1W": "BULL",
    }


def _actual_subset(actual: dict) -> dict:
    trace = actual.get("decision_trace", {})
    return {
        "regime_1D": actual.get("regime_1D"),
        "regime_1W": actual.get("regime_1W"),
        "score_raw": trace.get("score_raw"),
        "score_final": trace.get("score_final"),
        "base_action": trace.get("base_action"),
        "action_final": actual.get("action_final"),
        "confidence_cap": actual.get("confidence_cap"),
        "gates_triggered": actual.get("gates_triggered", []),
        "conflicts": actual.get("conflicts", []),
    }


# ---------------- Reference DRL implementation for semantic lock ----------------

def _reference_drl(policy: dict, inputs: dict[str, Any], now_iso: str) -> dict:
    required = list(policy["inputs"]["required_metrics"]) + list(policy["inputs"]["required_flags"]) + ["ticker", "as_of"]
    missing = [k for k in required if k not in inputs]
    if missing:
        raise IntegrityFailure(f"Reference DRL missing required inputs: {missing}")

    profile_name = str(policy["policy"]["default_profile"])
    profile = policy["actions"]["profiles"][profile_name]

    state: dict[str, Any] = dict(inputs)

    # 2) zones
    zones: dict[str, str] = {}
    for metric, zone_cfg in policy["normalization"]["zones"].items():
        zone = _ref_select_range_name(float(state[metric]), zone_cfg["ranges"])
        zone_key = f"{metric}_zone"
        zones[zone_key] = zone
        state[zone_key] = zone

    # 3) derived features
    derived: dict[str, Any] = {}
    for feature in policy["normalization"]["derived_features"]:
        val = _ref_eval_expr(str(feature["expression"]), state)
        derived[str(feature["name"])] = val
        state[str(feature["name"])] = val

    # 4) regimes
    regime_cfg = policy["states"]["regime"]
    for tf in regime_cfg["timeframes"]:
        bull_votes = 0
        bear_votes = 0

        for component in regime_cfg["components"]:
            feature_name = str(component["feature"]).replace("${tf}", str(tf))
            comp_value = state[feature_name]
            ctx = dict(state)
            ctx["value"] = comp_value

            if bool(_ref_eval_expr(str(component["bullish_when"]), ctx)):
                bull_votes += int(component.get("weight", 1))
            if bool(_ref_eval_expr(str(component["bearish_when"]), ctx)):
                bear_votes += int(component.get("weight", 1))

        if bull_votes >= int(regime_cfg["voting"]["bullish_min_votes"]) and bull_votes > bear_votes:
            regime_state = "BULL"
        elif bear_votes >= int(regime_cfg["voting"]["bearish_min_votes"]) and bear_votes > bull_votes:
            regime_state = "BEAR"
        else:
            regime_state = str(regime_cfg["voting"]["tie_state"])

        state[f"regime_{tf}"] = regime_state

    # 5) scoring
    score_components_total = 0.0
    for component in policy["scoring"]["components"]:
        subtotal = 0.0
        for contribution in component["contributions"]:
            if bool(_ref_eval_expr(str(contribution["when"]), state)):
                subtotal += float(contribution["score"])
        score_components_total += subtotal

    multiplier_total = 1.0
    for multiplier in policy["scoring"].get("multipliers", []):
        metric_value = float(state[str(multiplier["metric"])])
        multiplier_total *= _ref_select_range_value(metric_value, multiplier["bands"], "multiplier")

    penalty_total = 0.0
    for penalty in policy["scoring"].get("penalties", []):
        metric_value = float(state[str(penalty["metric"])])
        penalty_total += _ref_select_range_value(metric_value, penalty["bands"], "penalty")

    score_raw = score_components_total + penalty_total
    score_final = score_raw * multiplier_total
    state["score_raw"] = score_raw
    state["score_final"] = score_final

    # 6) base action from score bands
    score_bands = profile["score_bands"]
    base_band = _ref_resolve_score_band(score_final, score_bands)
    base_action = str(base_band["action"])
    action_current = base_action

    action_conf = {str(b["action"]): int(b["base_confidence"]) for b in score_bands}
    base_confidence = action_conf.get(base_action, int(base_band["base_confidence"]))

    # 7) governor rules
    disallowed_actions: set[str] = set()
    confidence_delta_total = 0
    for rule in profile.get("governor_rules", []):
        if bool(_ref_eval_expr(str(rule["when"]), state)):
            for action in rule.get("disallow_actions", []):
                disallowed_actions.add(str(action))
            confidence_delta_total += int(rule.get("confidence_cap_delta", 0))

    # 8) gates in order
    gates_triggered: list[str] = []
    conflicts: list[str] = []
    gate_defs = policy["gates"]["definitions"]
    for gate_id in policy["gates"]["evaluation_order"]:
        gate = gate_defs[gate_id]
        if not bool(_ref_eval_expr(str(gate["trigger_when"]), state)):
            continue

        gates_triggered.append(str(gate_id))
        effects = gate.get("effects", {})

        confidence_delta_total += int(effects.get("confidence_cap_delta", 0))

        for c in effects.get("add_conflicts", []):
            conflict = str(c)
            if conflict not in conflicts:
                conflicts.append(conflict)

        gate_disallow = [str(a) for a in effects.get("disallow_actions", [])]
        exception_cfg = gate.get("exceptions", {}).get("allow_action_when")
        if exception_cfg:
            exc_action = str(exception_cfg.get("action", ""))
            if exc_action and exc_action in gate_disallow:
                if bool(_ref_eval_expr(str(exception_cfg["when"]), state)):
                    gate_disallow = [a for a in gate_disallow if a != exc_action]

        for action in gate_disallow:
            disallowed_actions.add(action)

        force_action = effects.get("force_action")
        if force_action:
            action_current = str(force_action)

        if gate_id == "G_HIGH_VOL" and action_current != "WAIT":
            action_current = "WAIT"

    # 9) enforce disallowed actions
    if action_current in disallowed_actions:
        action_current = _ref_next_safe_action(disallowed_actions)

    confidence_base = action_conf.get(action_current, base_confidence)
    confidence_cap = confidence_base + confidence_delta_total

    stale = _ref_is_stale(str(inputs["as_of"]), now_iso)
    if stale:
        action_current = "WAIT"
        confidence_base = action_conf.get("WAIT", confidence_base)
        confidence_cap = min(confidence_base + confidence_delta_total, 55)
        if "STALE_DATA" not in conflicts:
            conflicts.append("STALE_DATA")

    confidence_cap = max(0, min(100, int(round(confidence_cap))))

    # 10) subset for comparison
    return {
        "regime_1D": state.get("regime_1D"),
        "regime_1W": state.get("regime_1W"),
        "score_raw": round(score_raw, 6),
        "score_final": round(score_final, 6),
        "base_action": base_action,
        "action_final": action_current,
        "confidence_cap": confidence_cap,
        "gates_triggered": gates_triggered,
        "conflicts": conflicts,
    }


def _ref_is_stale(as_of: str, now_iso: str, max_age_minutes: int = 90) -> bool:
    return (datetime.fromisoformat(now_iso) - datetime.fromisoformat(as_of)) > timedelta(minutes=max_age_minutes)


def _ref_in_range(value: float, min_v: float | None, max_v: float | None) -> bool:
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value >= max_v:
        return False
    return True


def _ref_select_range_name(value: float, ranges: list[dict[str, Any]]) -> str:
    for row in ranges:
        if _ref_in_range(value, row.get("min"), row.get("max")):
            return str(row["name"])
    raise IntegrityFailure(f"No zone matched value={value}")


def _ref_select_range_value(value: float, ranges: list[dict[str, Any]], key: str) -> float:
    for row in ranges:
        if _ref_in_range(value, row.get("min"), row.get("max")):
            return float(row[key])
    raise IntegrityFailure(f"No band matched value={value} key={key}")


def _ref_split_ternary(expr: str) -> tuple[str, str, str] | None:
    depth = 0
    in_single = False
    in_double = False
    q_pos = -1

    for i, ch in enumerate(expr):
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            q_pos = i
            break

    if q_pos == -1:
        return None

    cond = expr[:q_pos].strip()
    rest = expr[q_pos + 1 :]

    depth = 0
    in_single = False
    in_double = False
    nested = 0
    c_pos = -1
    for i, ch in enumerate(rest):
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            nested += 1
        elif ch == ":" and depth == 0:
            if nested == 0:
                c_pos = i
                break
            nested -= 1

    if c_pos == -1:
        raise IntegrityFailure(f"Malformed ternary expression: {expr}")

    return cond, rest[:c_pos].strip(), rest[c_pos + 1 :].strip()


class _RefSafeEval(ast.NodeVisitor):
    def __init__(self, ctx: dict[str, Any]):
        self.ctx = ctx

    def visit_Expression(self, node: ast.Expression) -> Any:  # noqa: N802
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:  # noqa: N802
        if node.id in self.ctx:
            return self.ctx[node.id]
        raise IntegrityFailure(f"Unknown expression symbol: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: N802
        return node.value

    def visit_List(self, node: ast.List) -> Any:  # noqa: N802
        return [self.visit(v) for v in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:  # noqa: N802
        return tuple(self.visit(v) for v in node.elts)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # noqa: N802
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise IntegrityFailure(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:  # noqa: N802
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(self.visit(v)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(self.visit(v)):
                    return True
            return False
        raise IntegrityFailure(f"Unsupported bool op: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:  # noqa: N802
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise IntegrityFailure(f"Unsupported comparator: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True

    def generic_visit(self, node: ast.AST) -> Any:
        raise IntegrityFailure(f"Unsupported expression node: {type(node).__name__}")


def _ref_eval_expr(expr: str, ctx: dict[str, Any]) -> Any:
    expr = expr.strip()
    ternary = _ref_split_ternary(expr)
    if ternary is not None:
        cond, true_expr, false_expr = ternary
        return _ref_eval_expr(true_expr, ctx) if bool(_ref_eval_expr(cond, ctx)) else _ref_eval_expr(false_expr, ctx)
    parsed = ast.parse(expr, mode="eval")
    return _RefSafeEval(ctx).visit(parsed)


def _ref_resolve_score_band(score: float, bands: list[dict[str, Any]]) -> dict[str, Any]:
    for band in bands:
        if _ref_in_range(score, band.get("min"), band.get("max")):
            return band
    raise IntegrityFailure(f"No score band for score={score}")


def _ref_next_safe_action(disallowed_actions: set[str]) -> str:
    for candidate in ["WAIT", "ACCUMULATE", "REDUCE"]:
        if candidate not in disallowed_actions:
            return candidate
    return "WAIT"


# ---------------- Helpers ----------------

def _load_policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8")) or {}


def _load_fixtures() -> list[dict]:
    data = yaml.safe_load(FIXTURES_PATH.read_text(encoding="utf-8")) or {}
    fixtures = data.get("fixtures", [])
    if not fixtures:
        raise IntegrityFailure("No fixtures found")
    return fixtures


def _fixture_by_id(fixture_id: str) -> dict:
    for fx in _load_fixtures():
        if fx.get("id") == fixture_id:
            return fx
    raise IntegrityFailure(f"Fixture not found: {fixture_id}")


def _run_invariants() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    inv_cases = {
        "INV1": {
            "ticker": "INV1",
            "as_of": NOW_ISO,
            "price_last": 311.0,
            "ema_50": 318.0,
            "sma_200": 242.0,
            "rsi_14": 28.0,
            "macd": -1.0,
            "macd_signal": 0.0,
            "stoch_k": 10.0,
            "adx_14": 16.0,
            "vroc_14": 5.0,
            "atr_pct": 3.5,
            "supertrend_dir_1D": "BEAR",
            "supertrend_dir_1W": "BEAR",
        },
        "INV2": {
            "ticker": "INV2",
            "as_of": NOW_ISO,
            "price_last": 340.0,
            "ema_50": 320.0,
            "sma_200": 250.0,
            "rsi_14": 74.0,
            "macd": 5.0,
            "macd_signal": 3.0,
            "stoch_k": 97.0,
            "adx_14": 34.0,
            "vroc_14": 22.0,
            "atr_pct": 2.0,
            "supertrend_dir_1D": "BULL",
            "supertrend_dir_1W": "BULL",
        },
        "INV3": {
            "ticker": "INV3",
            "as_of": NOW_ISO,
            "price_last": 305.0,
            "ema_50": 318.0,
            "sma_200": 250.0,
            "rsi_14": 40.0,
            "macd": -1.0,
            "macd_signal": 0.0,
            "stoch_k": 35.0,
            "adx_14": 22.0,
            "vroc_14": 10.0,
            "atr_pct": 3.0,
            "supertrend_dir_1D": "BEAR",
            "supertrend_dir_1W": "BULL",
        },
        "INV4": {
            "ticker": "INV4",
            "as_of": NOW_ISO,
            "price_last": 200.0,
            "ema_50": 210.0,
            "sma_200": 220.0,
            "rsi_14": 50.0,
            "macd": -0.5,
            "macd_signal": -0.2,
            "stoch_k": 50.0,
            "adx_14": 20.0,
            "vroc_14": 5.0,
            "atr_pct": 6.5,
            "supertrend_dir_1D": "BEAR",
            "supertrend_dir_1W": "BEAR",
        },
        "INV5": {
            "ticker": "INV5",
            "as_of": "2026-02-10T06:00:00-05:00",
            "price_last": 300.0,
            "ema_50": 295.0,
            "sma_200": 250.0,
            "rsi_14": 55.0,
            "macd": 1.0,
            "macd_signal": 0.8,
            "stoch_k": 50.0,
            "adx_14": 22.0,
            "vroc_14": 10.0,
            "atr_pct": 3.0,
            "supertrend_dir_1D": "BULL",
            "supertrend_dir_1W": "BULL",
        },
    }

    r1 = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inv_cases["INV1"], now_iso=NOW_ISO)
    checks.append(
        {
            "id": "INV1",
            "passed": r1["action_final"] != "ACCUMULATE" and r1["action_final"] == "WAIT",
            "actual": r1,
        }
    )

    r2 = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inv_cases["INV2"], now_iso=NOW_ISO)
    checks.append(
        {
            "id": "INV2",
            "passed": r2["action_final"] != "ACCUMULATE" and r2["action_final"] == "WAIT",
            "actual": r2,
        }
    )

    r3 = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inv_cases["INV3"], now_iso=NOW_ISO)
    checks.append(
        {
            "id": "INV3",
            "passed": r3["action_final"] == "WAIT" and "TIMEFRAME_CONFLICT" in r3.get("conflicts", []),
            "actual": r3,
        }
    )

    r4 = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inv_cases["INV4"], now_iso=NOW_ISO)
    checks.append(
        {
            "id": "INV4",
            "passed": float(r4["confidence_cap"]) <= 55.0,
            "actual": r4,
        }
    )

    r5 = evaluate_drl(policy_path=str(POLICY_PATH), inputs=inv_cases["INV5"], now_iso=NOW_ISO)
    checks.append(
        {
            "id": "INV5",
            "passed": r5["action_final"] == "WAIT" and "STALE_DATA" in r5.get("conflicts", []) and float(r5["confidence_cap"]) <= 55.0,
            "actual": r5,
        }
    )

    return checks


def _diff_dict(expected: dict, actual: dict) -> dict:
    if DeepDiff is not None:
        dd = DeepDiff(expected, actual, ignore_order=True)
        if hasattr(dd, "to_dict"):
            return dd.to_dict()
        return dict(dd)

    if expected == actual:
        return {}

    changed = {}
    for key in sorted(set(expected.keys()).union(actual.keys())):
        if expected.get(key) != actual.get(key):
            changed[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    return {"values_changed": changed}


if __name__ == "__main__":
    main()
