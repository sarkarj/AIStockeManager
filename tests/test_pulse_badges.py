from __future__ import annotations

from app.ui.components.pulse_badges import compute_pulse_badges


def _base_context_pack() -> dict:
    return {
        "meta": {
            "generated_at": "2026-02-11T12:00:00-05:00",
            "data_quality": {
                "prices": {
                    "as_of": "2026-02-11T11:40:00-05:00",
                    "now": "2026-02-11T12:00:00-05:00",
                    "age_minutes": 20.0,
                    "stale": False,
                    "stale_minutes_threshold": 90,
                },
                "indicators": {
                    "as_of": "2026-02-11T11:40:00-05:00",
                    "now": "2026-02-11T12:00:00-05:00",
                    "age_minutes": 20.0,
                    "stale": False,
                    "stale_minutes_threshold": 90,
                },
                "overall_stale": False,
                "notes": [],
            },
        },
        "prices": {
            "as_of": "2026-02-11T11:40:00-05:00",
            "bars": [
                {"ts": f"2026-02-11T{h:02d}:00:00-05:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0}
                for h in range(1, 13)
            ],
        },
    }


def test_no_hub_fresh_prices_age_only() -> None:
    context_pack = _base_context_pack()
    badges = compute_pulse_badges(context_pack)

    assert badges["age_minutes"] == 20
    assert badges["show_degraded"] is False
    assert badges["show_tool_down"] is False


def test_stale_prices_sets_degraded() -> None:
    context_pack = _base_context_pack()
    context_pack["meta"]["data_quality"]["prices"]["stale"] = True
    badges = compute_pulse_badges(context_pack)

    assert badges["show_degraded"] is True
    assert "STALE_DATA" in badges["reasons"]


def test_prices_fetch_failure_sets_tool_down() -> None:
    context_pack = _base_context_pack()
    context_pack["meta"]["data_quality"]["notes"] = ["TOOL_DOWN"]
    badges = compute_pulse_badges(context_pack)

    assert badges["show_tool_down"] is True
    assert badges["show_degraded"] is True


def test_negative_age_clamped_to_zero() -> None:
    context_pack = _base_context_pack()
    context_pack["meta"]["data_quality"]["prices"]["age_minutes"] = -8.0
    badges = compute_pulse_badges(context_pack)

    assert badges["age_minutes"] == 0
