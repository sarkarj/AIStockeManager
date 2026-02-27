from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.marketdata.quote_provider import QuoteProvider
from app.core.marketdata.latest_quote import latest_quote_from_series, normalize_quote
from app.core.marketdata.chart_fetcher import Bar, ChartSeries


def _daily_frame() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc),
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_200_000],
        },
        index=idx,
    )


def _intraday_frame() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            datetime(2026, 2, 11, 22, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 11, 22, 5, tzinfo=timezone.utc),
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "Open": [102.1, 102.5],
            "High": [102.4, 102.9],
            "Low": [102.0, 102.4],
            "Close": [102.3, 102.8],
            "Volume": [10_000, 12_000],
        },
        index=idx,
    )


def test_quote_provider_returns_close_and_after_hours(monkeypatch, tmp_path: Path) -> None:
    provider = QuoteProvider(cache_dir=str(tmp_path))

    def _history(*, ticker: str, period: str, interval: str, prepost: bool):
        _ = ticker, prepost
        if period == "5d" and interval == "1d":
            return _daily_frame()
        if period == "1d" and interval == "5m":
            return _intraday_frame()
        return pd.DataFrame()

    monkeypatch.setattr(provider, "_history_with_yfinance", _history)

    quote = provider.get_quote("AAPL")

    assert quote.source == "live"
    assert quote.close_price == 102.0
    assert quote.prev_close_price == 101.0
    assert quote.after_hours_price == 102.8
    assert "MISSING_CLOSE" not in quote.quality_flags
    assert "MISSING_AFTER_HOURS" not in quote.quality_flags


def test_quote_provider_missing_intraday_marks_missing_after_hours(monkeypatch, tmp_path: Path) -> None:
    provider = QuoteProvider(cache_dir=str(tmp_path))

    def _history(*, ticker: str, period: str, interval: str, prepost: bool):
        _ = ticker, prepost
        if period == "5d" and interval == "1d":
            return _daily_frame()
        return pd.DataFrame()

    monkeypatch.setattr(provider, "_history_with_yfinance", _history)

    quote = provider.get_quote("MSFT")

    assert quote.source == "live"
    assert quote.close_price == 102.0
    assert quote.after_hours_price is None
    assert "MISSING_AFTER_HOURS" in quote.quality_flags


def test_quote_provider_uses_stale_cache_on_live_failure(monkeypatch, tmp_path: Path) -> None:
    provider = QuoteProvider(cache_dir=str(tmp_path))
    close_key = "quote:close:NVDA:5d:1d:0"
    after_key = "quote:after_hours:NVDA:1d:5m:1"

    close_payload = {
        "as_of": "2026-02-11T00:00:00+00:00",
        "bars": [
            {
                "ts": "2026-02-10T00:00:00+00:00",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000.0,
            },
            {
                "ts": "2026-02-11T00:00:00+00:00",
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1100.0,
            },
        ],
    }
    after_payload = {
        "as_of": "2026-02-11T22:05:00+00:00",
        "bars": [
            {
                "ts": "2026-02-11T22:05:00+00:00",
                "open": 102.5,
                "high": 102.9,
                "low": 102.4,
                "close": 102.7,
                "volume": 8000.0,
            }
        ],
    }
    provider.cache.set(close_key, close_payload, ttl_seconds=-1)
    provider.cache.set(after_key, after_payload, ttl_seconds=-1)

    def _raise(*args, **kwargs):
        raise RuntimeError("network_down")

    monkeypatch.setattr(provider, "_history_with_yfinance", _raise)

    quote = provider.get_quote("NVDA")

    assert quote.source == "cache"
    assert quote.close_price == 102.0
    assert quote.after_hours_price == 102.7
    assert "STALE" in quote.quality_flags


def test_latest_quote_from_series_prefers_after_hours() -> None:
    bars = [
        Bar(
            ts=datetime(2026, 2, 11, 20, 55, tzinfo=timezone.utc),  # 15:55 ET RTH
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.8,
            volume=1000.0,
        ),
        Bar(
            ts=datetime(2026, 2, 11, 21, 0, tzinfo=timezone.utc),  # 16:00 ET close
            open=100.8,
            high=101.2,
            low=100.7,
            close=101.0,
            volume=1200.0,
        ),
        Bar(
            ts=datetime(2026, 2, 11, 22, 30, tzinfo=timezone.utc),  # 17:30 ET after-hours
            open=101.0,
            high=101.4,
            low=100.9,
            close=101.3,
            volume=900.0,
        ),
    ]
    series = ChartSeries(
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

    quote = latest_quote_from_series(
        series=series,
        now=datetime(2026, 2, 11, 23, 0, tzinfo=timezone.utc),
    )
    assert quote.close_price == 101.0
    assert quote.after_hours_price == 101.3
    assert quote.latest_price == 101.3
    assert "MISSING_CLOSE" not in quote.quality_flags
    assert "MISSING_AFTER_HOURS" not in quote.quality_flags


def test_latest_quote_missing_bars_flags() -> None:
    series = ChartSeries(
        bars=[],
        as_of=datetime(2026, 2, 11, 23, 0, tzinfo=timezone.utc),
        source="none",
        error="empty_live",
        quality_flags=set(),
        cache_path=".cache/charts/test.json",
        cache_age_minutes=None,
        cache_hit=False,
        stale_cache=False,
        attempts=1,
    )
    quote = latest_quote_from_series(series=series)
    assert quote.latest_price is None
    assert "MISSING_BARS" in quote.quality_flags


def test_normalize_quote_uses_after_hours_and_prev_close_baseline() -> None:
    normalized = normalize_quote(
        {
            "symbol": "GOOG",
            "currency": "USD",
            "close_price": 300.0,
            "close_ts": "2026-02-13T21:00:00+00:00",
            "prev_close_price": 295.0,
            "after_hours_price": 302.5,
            "after_hours_ts": "2026-02-13T22:15:00+00:00",
            "source": "live",
            "quality_flags": [],
            "error": None,
        },
        now=datetime(2026, 2, 14, 2, 0, tzinfo=timezone.utc),
    )

    assert normalized["latest_price"] == 302.5
    assert normalized["latest_source"] == "after_hours"
    assert normalized["today_change_abs"] == 5.0
    assert round(float(normalized["today_change_pct"]), 6) == round((5.0 / 295.0) * 100.0, 6)
    assert normalized["after_hours_change_abs"] == 2.5
    assert round(float(normalized["after_hours_change_pct"]), 6) == round((2.5 / 300.0) * 100.0, 6)


def test_normalize_quote_missing_prev_close_sets_quality_flag() -> None:
    normalized = normalize_quote(
        {
            "symbol": "AAPL",
            "close_price": 185.0,
            "close_ts": "2026-02-13T21:00:00+00:00",
            "prev_close_price": None,
            "after_hours_price": 184.8,
            "after_hours_ts": "2026-02-13T22:00:00+00:00",
            "source": "cache",
            "quality_flags": [],
            "error": None,
        }
    )
    assert normalized["today_change_abs"] is None
    assert normalized["today_change_pct"] is None
    assert "MISSING_PREV_CLOSE" in set(normalized["quality_flags"])
