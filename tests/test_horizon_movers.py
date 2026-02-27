from __future__ import annotations

from datetime import datetime, timezone

from app.core.marketdata.chart_fetcher import Bar, ChartSeries
from app.ui.components.horizon import _build_top_movers


class FakeMarketQuery:
    def __init__(self, weekly: dict[str, ChartSeries], intraday: dict[str, ChartSeries]):
        self.weekly = weekly
        self.intraday = intraday
        self.calls: list[tuple[str, str]] = []

    def chart_series(self, ticker: str, range_key: str) -> ChartSeries:
        symbol = str(ticker).upper()
        key = str(range_key).upper()
        self.calls.append((symbol, key))
        if key == "1W":
            return self.weekly[symbol]
        if key == "1D":
            return self.intraday[symbol]
        raise KeyError(key)


def _make_weekly_series(prev_close: float, regular_close: float, latest_close: float) -> ChartSeries:
    bars = [
        Bar(
            ts=datetime(2026, 2, 23, 21, 0, tzinfo=timezone.utc),  # 16:00 ET
            open=prev_close,
            high=prev_close,
            low=prev_close,
            close=prev_close,
            volume=1000.0,
        ),
        Bar(
            ts=datetime(2026, 2, 24, 21, 0, tzinfo=timezone.utc),  # 16:00 ET
            open=regular_close,
            high=regular_close,
            low=regular_close,
            close=regular_close,
            volume=1200.0,
        ),
        Bar(
            ts=datetime(2026, 2, 24, 23, 0, tzinfo=timezone.utc),  # 18:00 ET
            open=latest_close,
            high=latest_close,
            low=latest_close,
            close=latest_close,
            volume=800.0,
        ),
    ]
    return ChartSeries(
        bars=bars,
        as_of=bars[-1].ts,
        source="live",
        error=None,
        quality_flags=set(),
        cache_path=".cache/charts/test.json",
        cache_age_minutes=0.0,
        cache_hit=False,
        stale_cache=False,
        attempts=1,
    )


def _make_intraday_series(base: float) -> ChartSeries:
    bars = [
        Bar(ts=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc), open=base, high=base + 1, low=base - 1, close=base, volume=100.0),
        Bar(ts=datetime(2026, 2, 24, 16, 0, tzinfo=timezone.utc), open=base, high=base + 1, low=base - 1, close=base + 0.5, volume=120.0),
        Bar(ts=datetime(2026, 2, 24, 17, 0, tzinfo=timezone.utc), open=base, high=base + 1, low=base - 1, close=base + 0.2, volume=130.0),
    ]
    return ChartSeries(
        bars=bars,
        as_of=bars[-1].ts,
        source="live",
        error=None,
        quality_flags=set(),
        cache_path=".cache/charts/test.json",
        cache_age_minutes=0.0,
        cache_hit=False,
        stale_cache=False,
        attempts=1,
    )


def test_horizon_ranks_all_symbols_and_fetches_sparkline_only_top10() -> None:
    universe = [f"T{i:02d}" for i in range(20)]
    weekly = {
        ticker: _make_weekly_series(prev_close=100.0, regular_close=100.0 + idx, latest_close=100.0 + idx + 0.5)
        for idx, ticker in enumerate(universe, start=1)
    }
    intraday = {ticker: _make_intraday_series(base=100.0 + idx) for idx, ticker in enumerate(universe, start=1)}
    query = FakeMarketQuery(weekly=weekly, intraday=intraday)

    rows = _build_top_movers(
        universe=universe,
        market_query=query,
        limit=10,
        metric_mode="today_after",
    )

    assert len(rows) == 10
    assert sum(1 for _, key in query.calls if key == "1W") == 20
    assert sum(1 for _, key in query.calls if key == "1D") == 10


def test_horizon_metric_mode_regular_vs_today_after() -> None:
    universe = ["AAA", "BBB"]
    weekly = {
        "AAA": _make_weekly_series(prev_close=100.0, regular_close=110.0, latest_close=90.0),
        "BBB": _make_weekly_series(prev_close=100.0, regular_close=105.0, latest_close=120.0),
    }
    intraday = {
        "AAA": _make_intraday_series(base=110.0),
        "BBB": _make_intraday_series(base=105.0),
    }
    query_regular = FakeMarketQuery(weekly=weekly, intraday=intraday)
    query_today_after = FakeMarketQuery(weekly=weekly, intraday=intraday)

    regular_rows = _build_top_movers(
        universe=universe,
        market_query=query_regular,
        limit=2,
        metric_mode="regular",
    )
    today_after_rows = _build_top_movers(
        universe=universe,
        market_query=query_today_after,
        limit=2,
        metric_mode="today_after",
    )

    assert regular_rows[0]["ticker"] == "AAA"
    assert today_after_rows[0]["ticker"] == "BBB"
