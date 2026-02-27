from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.marketdata.chart_fetcher import ChartFetcher, range_mapping


def _cache_payload() -> dict:
    bars = [
        {
            "ts": "2026-02-11T14:00:00+00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.6,
            "volume": 1000.0,
        },
        {
            "ts": "2026-02-11T14:05:00+00:00",
            "open": 100.7,
            "high": 101.2,
            "low": 100.4,
            "close": 100.9,
            "volume": 900.0,
        },
    ]
    return {"as_of": bars[-1]["ts"], "bars": bars}


def _seed_stale_cache(fetcher: ChartFetcher, ticker: str, range_key: str) -> None:
    mapping = range_mapping(range_key)
    key = fetcher._cache_key(ticker=ticker, range_key=range_key, mapping=mapping)
    fetcher.cache.set(key, _cache_payload(), ttl_seconds=-1)


def test_live_raises_uses_cache(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))
    _seed_stale_cache(fetcher, ticker="AAPL", range_key="1D")

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fetcher, "_history_with_yfinance", _raise)
    series = fetcher.fetch_chart_series(ticker="AAPL", range_key="1D")

    assert series.source == "cache"
    assert len(series.bars) == 2
    assert "EMPTY_LIVE" in series.quality_flags
    assert "STALE_CACHE" in series.quality_flags
    assert series.error is not None


def test_live_empty_uses_cache(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))
    _seed_stale_cache(fetcher, ticker="MSFT", range_key="1D")

    monkeypatch.setattr(fetcher, "_history_with_yfinance", lambda **kwargs: pd.DataFrame())
    series = fetcher.fetch_chart_series(ticker="MSFT", range_key="1D")

    assert series.source == "cache"
    assert len(series.bars) == 2
    assert "EMPTY_LIVE" in series.quality_flags


def test_live_fail_and_no_cache_returns_none(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))

    def _raise(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(fetcher, "_history_with_yfinance", _raise)
    series = fetcher.fetch_chart_series(ticker="GOOG", range_key="1D")

    assert series.source == "none"
    assert series.bars == []
    assert "EMPTY_LIVE" in series.quality_flags
    assert len(series.bars) != 24


def test_sort_and_dedupe_on_live(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))

    idx = pd.to_datetime(
        [
            datetime(2026, 2, 11, 14, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 11, 13, 55, tzinfo=timezone.utc),
            datetime(2026, 2, 11, 14, 0, tzinfo=timezone.utc),
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 99.7, 101.0],
            "High": [101.0, 100.4, 101.5],
            "Low": [99.6, 99.2, 100.8],
            "Close": [100.5, 100.1, 101.2],
            "Volume": [1000, 1200, 1400],
        },
        index=idx,
    )
    monkeypatch.setattr(fetcher, "_history_with_yfinance", lambda **kwargs: frame)

    series = fetcher.fetch_chart_series(ticker="NVDA", range_key="1D")
    assert series.source == "live"
    assert len(series.bars) == 2
    assert "STALE" not in series.quality_flags
    assert "STALE_CACHE" not in series.quality_flags
    assert not any("STALE" in flag for flag in series.quality_flags)

    ts_values = [bar.ts for bar in series.bars]
    assert ts_values == sorted(ts_values)
    assert series.bars[-1].close == 101.2


def test_empty_cache_payload_treated_as_miss(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))
    mapping = range_mapping("1D")
    cache_key = fetcher._cache_key(ticker="PLUG", range_key="1D", mapping=mapping)
    fetcher.cache.set(cache_key, {"as_of": "2026-02-11T12:00:00+00:00", "bars": []}, ttl_seconds=3600)

    def _raise(*args, **kwargs):
        raise RuntimeError("live unavailable")

    monkeypatch.setattr(fetcher, "_history_with_yfinance", _raise)
    series = fetcher.fetch_chart_series(ticker="PLUG", range_key="1D")

    assert series.source == "none"
    assert series.cache_hit is False
    assert series.bars == []
    assert "MISSING" in series.quality_flags


def test_force_revalidate_bypasses_fresh_cache(monkeypatch, tmp_path: Path) -> None:
    fetcher = ChartFetcher(cache_dir=str(tmp_path))
    mapping = range_mapping("1D")
    cache_key = fetcher._cache_key(ticker="AAPL", range_key="1D", mapping=mapping)
    fetcher.cache.set(cache_key, _cache_payload(), ttl_seconds=3600)

    idx = pd.to_datetime([datetime(2026, 2, 11, 15, 0, tzinfo=timezone.utc)], utc=True)
    frame = pd.DataFrame(
        {
            "Open": [120.0],
            "High": [121.0],
            "Low": [119.0],
            "Close": [120.5],
            "Volume": [2000],
        },
        index=idx,
    )
    monkeypatch.setattr(fetcher, "_history_with_yfinance", lambda **kwargs: frame)

    cached_series = fetcher.fetch_chart_series(ticker="AAPL", range_key="1D")
    assert cached_series.source == "cache"
    assert cached_series.bars[-1].close == 100.9

    live_series = fetcher.fetch_chart_series(ticker="AAPL", range_key="1D", force_revalidate=True)
    assert live_series.source == "live"
    assert live_series.bars[-1].close == 120.5
