from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.marketdata.chart_fetcher import Bar, ChartSeries
from app.ui.viewmodels.brain_market_vm import build_brain_market_vm


def _series(
    closes: list[float],
    *,
    start: datetime,
    step_hours: int = 1,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    volumes: list[float | None] | None = None,
    source: str = "cache",
) -> ChartSeries:
    bars: list[Bar] = []
    for idx, close in enumerate(closes):
        ts = start + timedelta(hours=idx * step_hours)
        low = lows[idx] if lows is not None else close - 1.0
        high = highs[idx] if highs is not None else close + 1.0
        volume = volumes[idx] if volumes is not None else 1000.0 + idx
        bars.append(
            Bar(
                ts=ts,
                open=close - 0.5,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return ChartSeries(
        bars=bars,
        as_of=bars[-1].ts if bars else start,
        source=source if bars else "none",
        error=None if bars else "empty",
        quality_flags=set(),
        cache_path=".cache/charts/test.json",
        cache_age_minutes=0.0,
        cache_hit=bool(bars),
        stale_cache=False,
        attempts=1,
    )


class _FakeFetcher:
    def __init__(self, mapping):
        self.mapping = mapping

    def fetch_chart_series(self, ticker: str, range_key: str) -> ChartSeries:
        return self.mapping.get(range_key, self.mapping.get("default"))


def _context_pack(metrics: dict) -> dict:
    return {
        "meta": {"ticker": "AAPL", "hub": {"status": "missing", "mode": "DEGRADED", "hub_valid": False}},
        "indicators": {"metrics": metrics},
        "drl": {
            "result": {
                "action_final": "WAIT",
                "confidence_cap": 60,
                "regime_1D": "BULL",
                "regime_1W": "BULL",
                "decision_trace": {
                    "score_components": {
                        "regime_score": {"score": 3.0},
                        "momentum_score": {"score": 2.0},
                    }
                },
                "gates_triggered": [],
                "conflicts": [],
            }
        },
    }


def test_quote_stats_and_gauges_from_bars() -> None:
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    day_series = _series(
        [100.0, 101.0, 102.0],
        start=now - timedelta(hours=3),
        lows=[99.0, 100.5, 101.4],
        highs=[100.6, 101.7, 102.3],
        volumes=[1000.0, 2000.0, 3000.0],
    )
    year_series = _series(
        [88.0, 95.0, 120.0, 110.0],
        start=now - timedelta(days=250),
        step_hours=24,
        lows=[80.0, 92.0, 109.0, 108.0],
        highs=[90.0, 100.0, 130.0, 115.0],
    )
    selected_series = _series([99.0, 100.5, 101.5], start=now - timedelta(days=20), step_hours=24)
    fetcher = _FakeFetcher({"1D": day_series, "1Y": year_series, "1M": selected_series, "default": selected_series})

    vm = build_brain_market_vm(
        _context_pack(
            {
                "price_last": 110.0,
                "ema_50": 100.0,
                "sma_200": 90.0,
                "supertrend_dir_1D": "BULL",
                "supertrend_dir_1W": "BULL",
                "rsi_14": 60.0,
                "macd": 1.0,
                "macd_signal": 0.5,
                "stoch_k": 60.0,
                "atr_pct": 0.8,
                "adx_14": 30.0,
                "vroc_14": 10.0,
            }
        ),
        selected_range_key="1M",
        chart_fetcher=fetcher,
    )

    assert vm.quote.open == 99.5
    assert vm.quote.volume == 6000
    assert vm.quote.day_low == 99.0
    assert vm.quote.day_high == 102.3
    assert vm.quote.year_low == 80.0
    assert vm.quote.year_high == 130.0
    assert vm.price_display == 102.0

    gauges = {g.label: g for g in vm.gauges}
    assert gauges["TREND"].value == "BULLISH"
    assert gauges["TREND"].score == 82.0
    assert gauges["TREND"].tone == "good"
    assert gauges["MOMENTUM"].value == "BULLISH"
    assert gauges["MOMENTUM"].score == 80.0
    assert gauges["MOMENTUM"].tone == "good"
    assert gauges["RISK"].value == "LOW"
    assert gauges["RISK"].score == 20.0
    assert gauges["RISK"].tone == "good"
    assert gauges["STRENGTH"].value == "STRONG"
    assert gauges["STRENGTH"].score == 82.0
    assert gauges["STRENGTH"].tone == "good"


def test_missing_day_and_year_ranges_do_not_crash() -> None:
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    empty = _series([], start=now, source="none")
    selected_series = _series([5.0, 6.0], start=now - timedelta(days=2), step_hours=24)
    fetcher = _FakeFetcher({"1D": empty, "1Y": empty, "3M": selected_series, "default": selected_series})

    vm = build_brain_market_vm(
        _context_pack({}),
        selected_range_key="3M",
        chart_fetcher=fetcher,
    )

    assert vm.quote.open is None
    assert vm.quote.day_low is None
    assert vm.quote.day_high is None
    assert vm.quote.volume is None
    assert vm.quote.year_low is None
    assert vm.quote.year_high is None
    assert "DAY_RANGE_MISSING" in vm.quote.notes
    assert "YEAR_RANGE_MISSING" in vm.quote.notes

    gauges = {g.label: g for g in vm.gauges}
    assert gauges["TREND"].value == "—"
    assert gauges["MOMENTUM"].value == "—"
    assert gauges["RISK"].value == "—"
    assert gauges["STRENGTH"].value == "—"
