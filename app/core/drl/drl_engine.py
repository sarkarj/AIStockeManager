from __future__ import annotations

import ast
from datetime import datetime, timedelta
from typing import Any

import yaml


class PolicyValidationError(ValueError):
    """Raised when a DRL policy is malformed."""


def _load_policy(policy_path: str) -> dict[str, Any]:
    with open(policy_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise PolicyValidationError("Policy root must be a mapping")

    _validate_policy(raw)
    return raw


def _validate_policy(policy: dict[str, Any]) -> None:
    top_keys = ["policy", "inputs", "normalization", "states", "scoring", "gates", "actions", "output"]
    missing = [k for k in top_keys if k not in policy]
    if missing:
        raise PolicyValidationError(f"Policy missing top-level keys: {missing}")

    policy_info = policy["policy"]
    for key in ["id", "version", "default_profile"]:
        if key not in policy_info:
            raise PolicyValidationError(f"policy.{key} is required")

    default_profile = policy_info["default_profile"]
    profiles = policy["actions"].get("profiles", {})
    if default_profile not in profiles:
        raise PolicyValidationError(f"Default profile '{default_profile}' not found in actions.profiles")

    gate_order = policy["gates"].get("evaluation_order", [])
    gate_defs = policy["gates"].get("definitions", {})
    undefined_gates = [gate for gate in gate_order if gate not in gate_defs]
    if undefined_gates:
        raise PolicyValidationError(f"Undefined gates in evaluation_order: {undefined_gates}")

    required_metrics = policy["inputs"].get("required_metrics", [])
    required_flags = policy["inputs"].get("required_flags", [])
    if not required_metrics or not required_flags:
        raise PolicyValidationError("inputs.required_metrics and inputs.required_flags must be non-empty")


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception as exc:
        raise ValueError(f"Invalid ISO timestamp: {ts}") from exc


def _is_stale(as_of: str, now_iso: str, max_age_minutes: int = 90) -> bool:
    as_of_dt = _parse_iso(as_of)
    now_dt = _parse_iso(now_iso)
    return (now_dt - as_of_dt) > timedelta(minutes=max_age_minutes)


def _in_range(value: float, min_v: float | None, max_v: float | None) -> bool:
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value >= max_v:
        return False
    return True


def _select_range_name(value: float, ranges: list[dict[str, Any]]) -> str:
    for r in ranges:
        if _in_range(value, r.get("min"), r.get("max")):
            return str(r["name"])
    raise ValueError(f"No range matched value={value}")


def _select_range_value(value: float, ranges: list[dict[str, Any]], key: str) -> float:
    for r in ranges:
        if _in_range(value, r.get("min"), r.get("max")):
            return float(r[key])
    raise ValueError(f"No band matched value={value} for key={key}")


def _split_ternary(expr: str) -> tuple[str, str, str] | None:
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
        raise ValueError(f"Malformed ternary expression: {expr}")

    true_expr = rest[:c_pos].strip()
    false_expr = rest[c_pos + 1 :].strip()
    return cond, true_expr, false_expr


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, context: dict[str, Any]):
        self.context = context

    def visit_Expression(self, node: ast.Expression) -> Any:  # noqa: N802
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:  # noqa: N802
        if node.id in self.context:
            return self.context[node.id]
        raise ValueError(f"Unknown name in expression: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: N802
        return node.value

    def visit_List(self, node: ast.List) -> Any:  # noqa: N802
        return [self.visit(el) for el in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:  # noqa: N802
        return tuple(self.visit(el) for el in node.elts)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # noqa: N802
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:  # noqa: N802
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = bool(self.visit(value))
                if not result:
                    return False
            return bool(result)
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if bool(self.visit(value)):
                    return True
            return False
        raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:  # noqa: N802
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
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
                raise ValueError(f"Unsupported comparator: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _eval_python_expr(expr: str, context: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator(context)
    return evaluator.visit(tree)


def _eval_expr(expr: str, context: dict[str, Any]) -> Any:
    expr = expr.strip()
    ternary = _split_ternary(expr)
    if ternary is not None:
        cond, true_expr, false_expr = ternary
        return _eval_expr(true_expr, context) if bool(_eval_expr(cond, context)) else _eval_expr(false_expr, context)
    return _eval_python_expr(expr, context)


def _resolve_score_band(score: float, score_bands: list[dict[str, Any]]) -> dict[str, Any]:
    for band in score_bands:
        if _in_range(score, band.get("min"), band.get("max")):
            return band
    raise ValueError(f"No score band matched score={score}")


def _pick_next_safe_action(disallowed_actions: set[str]) -> str:
    for candidate in ["WAIT", "ACCUMULATE", "REDUCE"]:
        if candidate not in disallowed_actions:
            return candidate
    return "WAIT"


def evaluate_drl(policy_path: str, inputs: dict[str, Any], now_iso: str) -> dict[str, Any]:
    policy = _load_policy(policy_path)

    # 1) validate required inputs exist
    required_metrics = policy["inputs"]["required_metrics"]
    required_flags = policy["inputs"]["required_flags"]
    required = list(required_metrics) + list(required_flags) + ["ticker", "as_of"]
    missing = [k for k in required if k not in inputs]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    profile_name = policy["policy"]["default_profile"]
    profile = policy["actions"]["profiles"][profile_name]

    working: dict[str, Any] = dict(inputs)

    # 2) compute zones
    zones: dict[str, str] = {}
    for metric, cfg in policy["normalization"]["zones"].items():
        metric_value = float(working[metric])
        zone_name = _select_range_name(metric_value, cfg["ranges"])
        zone_key = f"{metric}_zone"
        zones[zone_key] = zone_name
        working[zone_key] = zone_name

    # 3) compute derived features
    derived_features: dict[str, Any] = {}
    for feat in policy["normalization"]["derived_features"]:
        value = _eval_expr(str(feat["expression"]), working)
        derived_features[feat["name"]] = value
        working[feat["name"]] = value

    # 4) compute regimes (1D/1W) by weighted voting
    regime_cfg = policy["states"]["regime"]
    regime_vote_trace: dict[str, Any] = {}

    for tf in regime_cfg["timeframes"]:
        bull_votes = 0
        bear_votes = 0
        component_votes: list[dict[str, Any]] = []

        for component in regime_cfg["components"]:
            feature_name = str(component["feature"]).replace("${tf}", tf)
            value = working[feature_name]
            local_ctx = dict(working)
            local_ctx["value"] = value

            is_bull = bool(_eval_expr(str(component["bullish_when"]), local_ctx))
            is_bear = bool(_eval_expr(str(component["bearish_when"]), local_ctx))
            weight = int(component.get("weight", 1))

            if is_bull:
                bull_votes += weight
            if is_bear:
                bear_votes += weight

            component_votes.append(
                {
                    "name": component["name"],
                    "feature": feature_name,
                    "value": value,
                    "weight": weight,
                    "bullish": is_bull,
                    "bearish": is_bear,
                }
            )

        if bull_votes >= int(regime_cfg["voting"]["bullish_min_votes"]) and bull_votes > bear_votes:
            regime_state = "BULL"
        elif bear_votes >= int(regime_cfg["voting"]["bearish_min_votes"]) and bear_votes > bull_votes:
            regime_state = "BEAR"
        else:
            regime_state = regime_cfg["voting"]["tie_state"]

        regime_key = f"regime_{tf}"
        working[regime_key] = regime_state
        regime_vote_trace[regime_key] = {
            "bull_votes": bull_votes,
            "bear_votes": bear_votes,
            "components": component_votes,
            "state": regime_state,
        }

    # 5) compute scoring + multiplier + penalty
    score_components: dict[str, Any] = {}
    score_component_total = 0.0

    for comp in policy["scoring"]["components"]:
        component_score = 0.0
        applied: list[str] = []
        for contribution in comp["contributions"]:
            if bool(_eval_expr(str(contribution["when"]), working)):
                component_score += float(contribution["score"])
                applied.append(str(contribution["name"]))
        score_components[comp["name"]] = {"score": component_score, "applied": applied}
        score_component_total += component_score

    multipliers: dict[str, float] = {}
    multiplier_total = 1.0
    for mult in policy["scoring"].get("multipliers", []):
        metric_val = float(working[mult["metric"]])
        val = _select_range_value(metric_val, mult["bands"], "multiplier")
        multipliers[mult["name"]] = val
        multiplier_total *= val

    penalties: dict[str, float] = {}
    penalty_total = 0.0
    for pen in policy["scoring"].get("penalties", []):
        metric_val = float(working[pen["metric"]])
        val = _select_range_value(metric_val, pen["bands"], "penalty")
        penalties[pen["name"]] = val
        penalty_total += val

    score_raw = score_component_total + penalty_total
    score_final = score_raw * multiplier_total
    working["score_raw"] = score_raw
    working["score_final"] = score_final

    # 6) map score to base action bands
    score_bands = profile["score_bands"]
    base_band = _resolve_score_band(score_final, score_bands)
    base_action = str(base_band["action"])
    action_current = base_action

    action_confidence_map: dict[str, int] = {}
    for band in score_bands:
        action_confidence_map[str(band["action"])] = int(band["base_confidence"])
    base_confidence = action_confidence_map.get(base_action, int(base_band["base_confidence"]))

    # 7) apply governor rules (weekly)
    disallowed_actions: set[str] = set()
    confidence_delta_total = 0
    governor_applied: list[str] = []

    for rule in profile.get("governor_rules", []):
        if bool(_eval_expr(str(rule["when"]), working)):
            governor_applied.append(str(rule["name"]))
            for action in rule.get("disallow_actions", []):
                disallowed_actions.add(str(action))
            confidence_delta_total += int(rule.get("confidence_cap_delta", 0))

    # 8) apply gates in strict evaluation order (with oversold exception rule)
    gates_triggered: list[str] = []
    conflicts: list[str] = []

    gate_defs = policy["gates"]["definitions"]
    for gate_id in policy["gates"]["evaluation_order"]:
        gate = gate_defs[gate_id]
        if not bool(_eval_expr(str(gate["trigger_when"]), working)):
            continue

        gates_triggered.append(str(gate_id))
        effects = gate.get("effects", {})

        confidence_delta_total += int(effects.get("confidence_cap_delta", 0))

        for conflict in effects.get("add_conflicts", []):
            c = str(conflict)
            if c not in conflicts:
                conflicts.append(c)

        gate_disallow = [str(a) for a in effects.get("disallow_actions", [])]

        exc = gate.get("exceptions", {}).get("allow_action_when")
        if exc:
            exc_action = str(exc.get("action", ""))
            if exc_action and exc_action in gate_disallow:
                if bool(_eval_expr(str(exc["when"]), working)):
                    gate_disallow = [a for a in gate_disallow if a != exc_action]

        for action in gate_disallow:
            disallowed_actions.add(action)

        force_action = effects.get("force_action")
        if force_action:
            action_current = str(force_action)

        if gate_id == "G_HIGH_VOL" and action_current != "WAIT":
            action_current = "WAIT"

    # 9) enforce disallowed actions deterministically (fallback to WAIT)
    if action_current in disallowed_actions:
        action_current = _pick_next_safe_action(disallowed_actions)

    confidence_base = action_confidence_map.get(action_current, base_confidence)
    confidence_cap = confidence_base + confidence_delta_total

    watch_conditions: list[str] = []

    stale = _is_stale(str(inputs["as_of"]), now_iso)
    if stale:
        action_current = "WAIT"
        confidence_base = action_confidence_map.get("WAIT", confidence_base)
        confidence_cap = min(confidence_base + confidence_delta_total, 55)
        if "STALE_DATA" not in conflicts:
            conflicts.append("STALE_DATA")

    confidence_cap = max(0, min(100, int(round(confidence_cap))))

    # 10) produce decision_trace
    decision_trace = {
        "policy_id": policy["policy"]["id"],
        "policy_version": policy["policy"]["version"],
        "ticker": inputs["ticker"],
        "profile": profile_name,
        "timestamp": now_iso,
        "inputs_used": {k: inputs[k] for k in required},
        "zones": zones,
        "derived_features": derived_features,
        "regime_1D": working["regime_1D"],
        "regime_1W": working["regime_1W"],
        "score_components": score_components,
        "multipliers": multipliers,
        "penalties": penalties,
        "score_raw": round(score_raw, 6),
        "score_final": round(score_final, 6),
        "base_action": base_action,
        "governor_applied": governor_applied,
        "gates_triggered": gates_triggered,
        "action_final": action_current,
        "confidence_base": confidence_base,
        "confidence_cap": confidence_cap,
        "conflicts": conflicts,
        "watch_conditions": watch_conditions,
        "regime_vote_trace": regime_vote_trace,
        "disallowed_actions": sorted(disallowed_actions),
    }

    required_trace_fields = policy["output"].get("decision_trace_required_fields", [])
    missing_trace = [k for k in required_trace_fields if k not in decision_trace]
    if missing_trace:
        raise RuntimeError(f"Decision trace missing required fields: {missing_trace}")

    return {
        "regime_1D": working["regime_1D"],
        "regime_1W": working["regime_1W"],
        "action_final": action_current,
        "confidence_cap": confidence_cap,
        "gates_triggered": gates_triggered,
        "conflicts": conflicts,
        "watch_conditions": watch_conditions,
        "decision_trace": decision_trace,
    }
