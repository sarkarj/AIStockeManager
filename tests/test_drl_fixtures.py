import pathlib
import yaml
import pytest

from app.core.drl.drl_engine import evaluate_drl

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES_PATH = ROOT / "app" / "data" / "fixtures" / "drl_fixtures.yaml"
POLICY_PATH = ROOT / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

NOW_ISO = "2026-02-11T12:00:00-05:00"

def load_fixtures():
    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    fixtures = data.get("fixtures", [])
    if not fixtures:
        raise ValueError("No fixtures found under top-level key: fixtures")
    return fixtures

@pytest.mark.parametrize("fx", load_fixtures(), ids=lambda fx: fx.get("id", "UNKNOWN_FIXTURE"))
def test_drl_fixture(fx):
    inputs = fx["inputs"]
    expected = fx["expected"]

    result = evaluate_drl(
        policy_path=str(POLICY_PATH),
        inputs=inputs,
        now_iso=NOW_ISO,
    )

    for k in ["regime_1D", "regime_1W", "action_final", "confidence_cap", "gates_triggered", "conflicts", "decision_trace"]:
        assert k in result, f"Missing key '{k}' in DRL result for fixture {fx['id']}"

    assert result["regime_1D"] == expected["regime_1D"], f"{fx['id']}: regime_1D mismatch"
    assert result["regime_1W"] == expected["regime_1W"], f"{fx['id']}: regime_1W mismatch"
    assert result["action_final"] == expected["action_final"], f"{fx['id']}: action_final mismatch"

    assert result["confidence_cap"] <= expected["confidence_cap_max"], (
        f"{fx['id']}: confidence_cap {result['confidence_cap']} exceeds max {expected['confidence_cap_max']}"
    )

    expected_gates = set(expected.get("gates_triggered", []))
    actual_gates = set(result.get("gates_triggered", []))
    assert expected_gates.issubset(actual_gates), (
        f"{fx['id']}: expected gates {sorted(expected_gates)} not subset of actual gates {sorted(actual_gates)}"
    )

    expected_conflicts = set(expected.get("conflicts_contains", []))
    actual_conflicts = set(result.get("conflicts", []))
    assert expected_conflicts.issubset(actual_conflicts), (
        f"{fx['id']}: expected conflicts {sorted(expected_conflicts)} not subset of actual conflicts {sorted(actual_conflicts)}"
    )

    trace = result["decision_trace"]
    for k in ["policy_id", "policy_version", "timestamp", "ticker", "score_final", "base_action", "action_final"]:
        assert k in trace, f"{fx['id']}: decision_trace missing '{k}'"
