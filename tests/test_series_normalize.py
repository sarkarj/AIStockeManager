from __future__ import annotations

from app.core.market.series_normalize import normalize_bars_for_chart


def test_normalize_sorts_and_dedupes() -> None:
    bars = [
        {"ts": "2026-02-11T12:00:00Z", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 10},
        {"ts": "2026-02-11T10:00:00Z", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 8},
        {"ts": "2026-02-11T12:00:00Z", "open": 102, "high": 104, "low": 101, "close": 103, "volume": 12},
        {"ts": "2026-02-11T11:00:00Z", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 9},
    ]

    frame = normalize_bars_for_chart(bars)

    assert len(frame) == 3
    assert list(frame["close"]) == [100.0, 101.0, 103.0]
    assert frame["ts"].is_monotonic_increasing


def test_normalize_parses_string_timestamps() -> None:
    bars = [
        {"ts": "2026-02-11T09:30:00-05:00", "open": "200", "high": "202", "low": "199", "close": "201", "volume": "1000"},
        {"ts": "2026-02-11T10:30:00-05:00", "open": "201", "high": "203", "low": "200", "close": "202", "volume": "1200"},
    ]

    frame = normalize_bars_for_chart(bars)

    assert len(frame) == 2
    assert str(frame["ts"].dtype).startswith("datetime64[ns, UTC]")
    assert list(frame["close"]) == [201.0, 202.0]
