from __future__ import annotations

from app.ui.components.pulse import _downsample_bars, _sparkline_points_from_bars
from app.ui.viewmodels.brain_vm import build_brain_view_model
from app.ui.viewmodels.pulse_vm import build_pulse_row_vm


def _context_pack() -> dict:
    return {
        "meta": {
            "ticker": "GOOG",
            "data_quality": {
                "prices": {"age_minutes": 5.0, "stale": False},
                "overall_stale": False,
                "notes": [],
            },
        },
        "prices": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "bars": [
                {"ts": "2026-02-11T11:00:00-05:00", "close": 278.0},
                {"ts": "2026-02-11T12:00:00-05:00", "close": 279.0},
            ],
        },
        "indicators": {"metrics": {"price_last": 305.0}},
        "drl": {
            "result": {
                "action_final": "WAIT",
                "confidence_cap": 60,
                "decision_trace": {"ticker": "GOOG"},
            }
        },
    }


def _quote() -> dict:
    return {
        "symbol": "GOOG",
        "currency": "USD",
        "close_price": 278.12,
        "close_ts": "2026-02-11T21:00:00+00:00",
        "close_ts_local": "2026-02-11 16:00 ET",
        "prev_close_price": 260.21,
        "after_hours_price": 279.01,
        "after_hours_ts": "2026-02-11T22:30:00+00:00",
        "after_hours_ts_local": "2026-02-11 17:30 ET",
        "latest_price": 279.01,
        "latest_ts": "2026-02-11T22:30:00+00:00",
        "latest_ts_local": "2026-02-11 17:30 ET",
        "latest_source": "after_hours",
        "today_change_abs": 18.80,
        "today_change_pct": 7.225625,
        "after_hours_change_abs": 0.89,
        "after_hours_change_pct": 0.319979,
        "source": "live",
        "quality_flags": [],
        "error": None,
    }


def test_pulse_and_brain_use_same_canonical_quote_latest() -> None:
    context_pack = _context_pack()
    quote = _quote()
    series = {
        "bars": [
            {"ts": "2026-02-11T11:55:00-05:00", "close": 278.80},
            {"ts": "2026-02-11T12:00:00-05:00", "close": 279.00},
        ]
    }

    pulse_vm = build_pulse_row_vm(
        holding={"ticker": "GOOG", "quantity": 10.0, "avg_cost": 250.0},
        context_pack=context_pack,
        quote=quote,
        primary_series_close=279.00,
    )
    brain_vm = build_brain_view_model(
        context_pack=context_pack,
        quote=quote,
        series_for_selected_range=series,
    )

    assert pulse_vm["last_price"] == 279.01
    assert brain_vm["last_price"] == 279.01
    assert pulse_vm["quote"]["close_price"] == 278.12
    assert brain_vm["quote"]["close_price"] == 278.12
    assert pulse_vm["today_abs"] == 18.80
    assert pulse_vm["today_pct"] == 7.225625
    assert pulse_vm["quote"]["latest_source"] == "after_hours"
    assert brain_vm["quote"]["latest_source"] == "after_hours"


def test_pulse_sparkline_downsamples_real_1d_bars() -> None:
    bars = [
        {
            "ts": f"2026-02-11T10:{i:02d}:00-05:00",
            "open": 100.0 + i * 0.1,
            "high": 100.1 + i * 0.1,
            "low": 99.9 + i * 0.1,
            "close": 100.0 + i * 0.1,
            "volume": 1000.0 + i,
        }
        for i in range(120)
    ]
    downsampled = _downsample_bars(bars, max_points=60)
    points = _sparkline_points_from_bars(downsampled, n=60)

    assert len(points) <= 60
    assert points[0][1] == bars[0]["close"]
    assert points[-1][1] == bars[-1]["close"]
