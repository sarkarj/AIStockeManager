from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.marketdata.chart_fetcher import Bar, ChartSeries
from app.core.marketdata.query_graph import MarketQueryService


def test_market_query_short_and_long_context_contracts() -> None:
    calls: list[dict[str, Any]] = []

    def loader(*, ticker: str, generate_hub_card: bool, interval: str, lookback_days: int) -> dict[str, Any]:
        calls.append(
            {
                "ticker": ticker,
                "generate_hub_card": generate_hub_card,
                "interval": interval,
                "lookback_days": lookback_days,
            }
        )
        return {"meta": {"ticker": ticker}, "drl": {"result": {"action_final": "WAIT", "confidence_cap": 0.0}}}

    query = MarketQueryService(
        cache_dir=".cache/charts",
        context_loader=loader,
        short_interval="1h",
        short_lookback_days=30,
        long_interval="1d",
        long_lookback_days=365,
    )

    short_pack = query.short_context("goog")
    long_pack = query.long_context("goog", generate_hub_card=True)

    assert short_pack["meta"]["query_mode"] == "short"
    assert long_pack["meta"]["query_mode"] == "long"
    assert short_pack["meta"]["query_contract"] == {
        "interval": "1h",
        "lookback_days": 30,
        "generate_hub_card": False,
    }
    assert long_pack["meta"]["query_contract"] == {
        "interval": "1d",
        "lookback_days": 365,
        "generate_hub_card": True,
    }
    assert calls == [
        {"ticker": "GOOG", "generate_hub_card": False, "interval": "1h", "lookback_days": 30},
        {"ticker": "GOOG", "generate_hub_card": True, "interval": "1d", "lookback_days": 365},
    ]


def test_market_query_context_cache_keys_include_query_mode() -> None:
    call_count = 0

    def loader(*, ticker: str, generate_hub_card: bool, interval: str, lookback_days: int) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"meta": {"ticker": ticker, "interval": interval, "lookback_days": lookback_days}}

    query = MarketQueryService(cache_dir=".cache/charts", context_loader=loader)

    query.short_context("NVDA")
    query.short_context("NVDA")
    query.long_context("NVDA", generate_hub_card=True)
    query.long_context("NVDA", generate_hub_card=True)
    query.long_context("NVDA", generate_hub_card=False)
    query.long_context("NVDA", generate_hub_card=False)

    assert call_count == 3


def test_market_query_revalidate_tickers_collects_stats(monkeypatch) -> None:
    def loader(*, ticker: str, generate_hub_card: bool, interval: str, lookback_days: int) -> dict[str, Any]:
        return {"meta": {"ticker": ticker}}

    query = MarketQueryService(cache_dir=".cache/charts", context_loader=loader)

    def _fake_fetch(*, ticker: str, range_key: str, force_revalidate: bool = False):
        assert force_revalidate is True
        source = "live" if str(ticker).upper() == "AAPL" else "cache"
        return ChartSeries(
            bars=[
                Bar(
                    ts=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
                    open=100.0,
                    high=101.0,
                    low=99.5,
                    close=100.5,
                    volume=1000.0,
                )
            ],
            as_of=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
            source=source,
            error=None,
            quality_flags=set(),
            cache_path=".cache/charts/test.json",
            cache_age_minutes=0.0,
            cache_hit=(source == "cache"),
            stale_cache=False,
            attempts=1,
        )

    monkeypatch.setattr(query._fetcher, "fetch_chart_series", _fake_fetch)
    stats = query.revalidate_tickers(tickers={"AAPL", "MSFT"}, range_keys=("1D",))

    assert stats == {"attempted": 2, "live": 1, "cache": 1, "none": 0, "errors": 0}
