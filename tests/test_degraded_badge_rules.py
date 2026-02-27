from __future__ import annotations

from app.ui.components.trust_badges import compute_brain_trust_state


def test_no_degraded_when_live_and_hub_normal() -> None:
    context_pack = {
        "meta": {
            "hub": {"status": "present", "mode": "NORMAL", "hub_valid": True, "reason": ""},
            "data_quality": {"overall_stale": False, "notes": []},
        },
        "hub_card": {
            "drivers": [{"text": "d", "citations": ["indicator:rsi_14"]}],
            "conflicts": [],
            "watch": [{"text": "w", "citations": ["indicator:rsi_14"]}],
            "evidence": {"used_ids": ["indicator:rsi_14"]},
        },
    }
    state = compute_brain_trust_state(
        context_pack=context_pack,
        chart_series={"source": "live", "flags": [], "diagnostics": {"error": None}},
        market_data_provider_up=True,
        quote={"source": "live", "quality_flags": [], "close_price": 100.0, "after_hours_price": 101.0},
        price_sanity_flags=[],
    )
    assert state["grounded"] is True
    assert state["degraded"] is False
    assert state["tool_down"] is False
