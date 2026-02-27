from __future__ import annotations

import pytest

from app.ui.viewmodels.brain_vm import build_brain_view_model
from app.ui.viewmodels.pulse_vm import build_pulse_row_vm


def _context_pack(action_final: str, confidence_cap: float, with_hub: bool = True) -> dict:
    pack = {
        "meta": {
            "ticker": "AAPL",
            "data_quality": {
                "prices": {"age_minutes": 10.0, "stale": False},
                "overall_stale": False,
                "notes": [],
            },
        },
        "prices": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "bars": [
                {"ts": "2026-02-11T11:00:00-05:00", "close": 99.0},
                {"ts": "2026-02-11T12:00:00-05:00", "close": 100.0},
            ],
        },
        "drl": {
            "result": {
                "action_final": action_final,
                "confidence_cap": confidence_cap,
                "decision_trace": {"ticker": "AAPL"},
            }
        },
        "indicators": {"metrics": {"rsi_14": 55.0}},
    }
    if with_hub:
        pack["hub_card"] = {
            "meta": {"mode": "FULL"},
            "summary": {
                "action_final": "WAIT",
                "confidence_cap": 10,
                "one_liner": "Hub summary text.",
            },
            "drivers": [{"text": "Driver", "citations": ["indicator:rsi_14"]}],
            "conflicts": [],
            "watch": [{"text": "Watch", "citations": ["indicator:rsi_14"]}],
            "evidence": {"used_ids": ["indicator:rsi_14"]},
        }
    return pack


def test_action_label_and_confidence_are_from_drl_only() -> None:
    context_pack = _context_pack(action_final="ACCUMULATE", confidence_cap=75.0, with_hub=True)
    vm = build_brain_view_model(context_pack)

    assert vm["ui_action_label"] == "BUY"
    assert vm["drl_action_raw"] == "ACCUMULATE"
    assert vm["confidence_cap"] == 75.0
    assert vm["one_liner"] == "Hub summary text."
    assert vm["hub_mode"] == "FULL"


def test_degraded_missing_hub_does_not_crash() -> None:
    context_pack = _context_pack(action_final="WAIT", confidence_cap=55.0, with_hub=False)
    context_pack["meta"]["data_quality"]["overall_stale"] = True
    context_pack["meta"]["data_quality"]["notes"] = ["BEDROCK_UNAVAILABLE: fallback"]

    vm = build_brain_view_model(context_pack)

    assert vm["ui_action_label"] == "HOLD"
    assert vm["confidence_cap"] == 55.0
    assert vm["one_liner"] is None
    assert vm["drivers"] == []
    assert vm["conflicts"] == []
    assert vm["watch"] == []


@pytest.mark.parametrize(
    ("action_final", "expected_label"),
    [
        ("ACCUMULATE", "BUY"),
        ("WAIT", "HOLD"),
        ("REDUCE", "SELL"),
    ],
)
def test_buy_hold_sell_mapping(action_final: str, expected_label: str) -> None:
    vm = build_brain_view_model(_context_pack(action_final=action_final, confidence_cap=70.0))
    assert vm["ui_action_label"] == expected_label


@pytest.mark.parametrize(
    ("action_final", "expected_label"),
    [
        ("ACCUMULATE", "BUY"),
        ("WAIT", "HOLD"),
        ("REDUCE", "SELL"),
    ],
)
def test_horizon_and_pulse_vms_use_same_mapping(action_final: str, expected_label: str) -> None:
    context_pack = _context_pack(action_final=action_final, confidence_cap=66.0, with_hub=False)
    holding = {"ticker": "AAPL", "avg_cost": 90.0, "quantity": 2.0}

    brain_vm = build_brain_view_model(context_pack)
    pulse_vm = build_pulse_row_vm(holding=holding, context_pack=context_pack)

    assert brain_vm["ui_action_label"] == expected_label
    assert pulse_vm["ui_action_label"] == expected_label
    assert brain_vm["confidence_cap"] == 66.0
    assert pulse_vm["confidence_cap"] == 66.0
