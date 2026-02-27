from __future__ import annotations

from app.ui.components.trust_badges import compute_brain_trust_state


def _healthy_context_pack() -> dict:
    return {
        "meta": {
            "hub": {
                "status": "present",
                "mode": "NORMAL",
                "hub_valid": True,
            },
            "data_quality": {
                "overall_stale": False,
                "notes": [],
            },
        },
        "hub_card": {
            "drivers": [{"text": "Driver", "citations": ["indicator:rsi_14"]}],
            "conflicts": [],
            "watch": [{"text": "Watch", "citations": ["indicator:rsi_14"]}],
            "evidence": {"used_ids": ["indicator:rsi_14"]},
        },
    }


def test_brain_badges_healthy_is_grounded_only() -> None:
    state = compute_brain_trust_state(
        context_pack=_healthy_context_pack(),
        chart_series={
            "source": "live",
            "flags": [],
            "diagnostics": {"error": None},
        },
        market_data_provider_up=True,
    )

    assert state["grounded"] is True
    assert state["degraded"] is False
    assert state["tool_down"] is False


def test_brain_badges_tool_down_when_no_chart_data() -> None:
    state = compute_brain_trust_state(
        context_pack=_healthy_context_pack(),
        chart_series={
            "source": "none",
            "flags": ["EMPTY_LIVE"],
            "diagnostics": {"error": "empty_live"},
        },
        market_data_provider_up=False,
    )

    assert state["tool_down"] is True
    assert state["degraded"] is True


def test_brain_badges_hub_degraded_no_grounded() -> None:
    context_pack = _healthy_context_pack()
    context_pack["meta"]["hub"]["mode"] = "DEGRADED"
    state = compute_brain_trust_state(
        context_pack=context_pack,
        chart_series={
            "source": "live",
            "flags": [],
            "diagnostics": {"error": None},
        },
        market_data_provider_up=True,
    )
    assert state["grounded"] is False
    assert state["degraded"] is True
    assert state["tool_down"] is False


def test_brain_badges_provider_down_sets_tool_down() -> None:
    context_pack = _healthy_context_pack()
    state = compute_brain_trust_state(
        context_pack=context_pack,
        chart_series={
            "source": "cache",
            "flags": ["STALE_CACHE"],
            "diagnostics": {"error": None},
        },
        market_data_provider_up=False,
    )
    assert state["tool_down"] is True
