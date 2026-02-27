from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.drl.drl_engine import evaluate_drl  # noqa: E402

POLICY_PATH = ROOT / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"
FIXTURES_PATH = ROOT / "app" / "data" / "fixtures" / "drl_fixtures.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print DRL decision_trace from a fixture ID or explicit metrics payload")
    parser.add_argument("--fixture-id", default=None, help="Fixture ID from drl_fixtures.yaml")
    parser.add_argument("--metrics-json", default=None, help="Raw JSON string for inputs (must include required DRL fields)")
    parser.add_argument("--metrics-file", default=None, help="Path to JSON file with DRL inputs")
    parser.add_argument("--ticker", default=None, help="Optional ticker override when using metrics payload")
    parser.add_argument("--now-iso", default=None, help="now_iso for DRL evaluation")
    parser.add_argument("--policy-path", default=str(POLICY_PATH), help="Path to DRL policy yaml")
    args = parser.parse_args()

    if not args.fixture_id and not args.metrics_json and not args.metrics_file:
        raise SystemExit("Provide --fixture-id OR --metrics-json/--metrics-file")

    if args.fixture_id:
        inputs = _inputs_from_fixture(args.fixture_id)
        now_iso = args.now_iso or inputs.get("as_of")
    else:
        inputs = _inputs_from_metrics(metrics_json=args.metrics_json, metrics_file=args.metrics_file)
        if args.ticker:
            inputs["ticker"] = args.ticker.strip().upper()
        if "ticker" not in inputs:
            raise SystemExit("metrics payload must include ticker or provide --ticker")
        if "as_of" not in inputs:
            raise SystemExit("metrics payload must include as_of")
        now_iso = args.now_iso or str(inputs["as_of"])

    result = evaluate_drl(policy_path=str(args.policy_path), inputs=inputs, now_iso=str(now_iso))
    trace = result.get("decision_trace", {})

    print("=" * 80)
    print(f"Ticker: {inputs.get('ticker')} | now_iso: {now_iso}")
    print(f"Action Final: {result.get('action_final')} | Confidence Cap: {result.get('confidence_cap')}")
    print("=" * 80)

    print("[STEP 1] Inputs Used")
    print(json.dumps(trace.get("inputs_used", {}), indent=2, ensure_ascii=True, sort_keys=True))

    print("\n[STEP 2] Zones")
    print(json.dumps(trace.get("zones", {}), indent=2, ensure_ascii=True, sort_keys=True))

    print("\n[STEP 3] Derived Features")
    print(json.dumps(trace.get("derived_features", {}), indent=2, ensure_ascii=True, sort_keys=True))

    print("\n[STEP 4] Regimes")
    print(json.dumps({"regime_1D": trace.get("regime_1D"), "regime_1W": trace.get("regime_1W")}, indent=2, ensure_ascii=True))

    print("\n[STEP 5] Score Components / Multipliers / Penalties")
    print(json.dumps(trace.get("score_components", {}), indent=2, ensure_ascii=True, sort_keys=True))
    print(json.dumps({"multipliers": trace.get("multipliers", {}), "penalties": trace.get("penalties", {})}, indent=2, ensure_ascii=True, sort_keys=True))
    print(json.dumps({"score_raw": trace.get("score_raw"), "score_final": trace.get("score_final")}, indent=2, ensure_ascii=True))

    print("\n[STEP 6] Base Action")
    print(json.dumps({"base_action": trace.get("base_action")}, indent=2, ensure_ascii=True))

    print("\n[STEP 7] Governor")
    print(json.dumps({"governor_applied": trace.get("governor_applied", [])}, indent=2, ensure_ascii=True))

    print("\n[STEP 8] Gates")
    print(json.dumps({"gates_triggered": trace.get("gates_triggered", []), "conflicts": trace.get("conflicts", [])}, indent=2, ensure_ascii=True))

    print("\n[STEP 9] Final Enforcement")
    print(json.dumps({"action_final": trace.get("action_final"), "confidence_cap": trace.get("confidence_cap")}, indent=2, ensure_ascii=True))

    print("\n[STEP 10] Full decision_trace")
    print(json.dumps(trace, indent=2, ensure_ascii=True, sort_keys=True))


def _inputs_from_fixture(fixture_id: str) -> dict:
    data = yaml.safe_load(FIXTURES_PATH.read_text(encoding="utf-8")) or {}
    fixtures = data.get("fixtures", [])
    for fx in fixtures:
        if fx.get("id") == fixture_id:
            return dict(fx["inputs"])
    raise SystemExit(f"Fixture not found: {fixture_id}")


def _inputs_from_metrics(metrics_json: str | None, metrics_file: str | None) -> dict:
    if metrics_file:
        payload_text = Path(metrics_file).read_text(encoding="utf-8")
        data = json.loads(payload_text)
        if not isinstance(data, dict):
            raise SystemExit("metrics file must contain a JSON object")
        return dict(data)

    if metrics_json:
        data = json.loads(metrics_json)
        if not isinstance(data, dict):
            raise SystemExit("metrics json must be a JSON object")
        return dict(data)

    raise SystemExit("No metrics payload provided")


if __name__ == "__main__":
    main()
