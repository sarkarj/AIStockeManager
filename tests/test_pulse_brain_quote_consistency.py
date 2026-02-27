from __future__ import annotations

from app.ui.viewmodels.brain_vm import build_brain_view_model
from app.ui.viewmodels.pulse_vm import build_pulse_row_vm


def test_pulse_brain_quote_consistent_latest_and_today() -> None:
    context_pack = {
        "meta": {"ticker": "GOOG", "data_quality": {"prices": {"age_minutes": 1.0}, "overall_stale": False, "notes": []}},
        "prices": {"as_of": "2026-02-11T12:00:00-05:00", "bars": [{"ts": "2026-02-11T12:00:00-05:00", "close": 279.0}]},
        "indicators": {"metrics": {"price_last": 279.0}},
        "drl": {"result": {"action_final": "WAIT", "confidence_cap": 50, "decision_trace": {"ticker": "GOOG"}}},
    }
    quote = {
        "symbol": "GOOG",
        "close_price": 278.12,
        "prev_close_price": 260.21,
        "after_hours_price": 279.01,
        "close_ts": "2026-02-11T21:00:00+00:00",
        "after_hours_ts": "2026-02-11T22:30:00+00:00",
        "latest_price": 279.01,
        "today_change_abs": 17.91,
        "today_change_pct": 6.882901502632489,
        "after_hours_change_abs": 0.89,
        "after_hours_change_pct": 0.31997700309143713,
        "quality_flags": [],
        "source": "live",
    }

    pulse_vm = build_pulse_row_vm(
        holding={"ticker": "GOOG", "quantity": 10.0, "avg_cost": 250.0},
        context_pack=context_pack,
        quote=quote,
        primary_series_close=279.0,
    )
    brain_vm = build_brain_view_model(
        context_pack=context_pack,
        quote=quote,
        series_for_selected_range={"bars": [{"ts": "2026-02-11T12:00:00-05:00", "close": 279.0}]},
    )

    assert pulse_vm["last_price"] == brain_vm["last_price"] == 279.01
    assert pulse_vm["today_abs"] == 17.91
    assert pulse_vm["today_pct"] == 6.882901502632489
