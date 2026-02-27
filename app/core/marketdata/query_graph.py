from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.marketdata.chart_fetcher import ChartFetcher, ChartSeries
from app.core.marketdata.quotes import QuoteSnapshot, get_quote_snapshot, quote_snapshot_to_dict

ContextLoader = Callable[..., dict[str, Any]]


@dataclass
class MarketQueryService:
    """Per-rerun market query coordinator to dedupe chart/quote fetches."""

    cache_dir: str = ".cache/charts"
    context_loader: ContextLoader | None = None
    short_interval: str = "1h"
    short_lookback_days: int = 60
    long_interval: str = "1h"
    long_lookback_days: int = 60
    cache_only: bool | None = None
    _fetcher: ChartFetcher = field(init=False, repr=False)
    _series_cache: dict[tuple[str, str], ChartSeries] = field(default_factory=dict, init=False, repr=False)
    _quote_cache: dict[str, QuoteSnapshot] = field(default_factory=dict, init=False, repr=False)
    _context_cache: dict[tuple[str, str, bool], dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetcher = ChartFetcher(cache_dir=self.cache_dir, stale_first=True, cache_only=self.cache_only)

    def chart_series(self, ticker: str, range_key: str) -> ChartSeries:
        ticker_norm = str(ticker or "").strip().upper()
        range_norm = str(range_key or "").strip().upper()
        key = (ticker_norm, range_norm)
        if key not in self._series_cache:
            self._series_cache[key] = self._fetcher.fetch_chart_series(ticker=ticker_norm, range_key=range_norm)
        return self._series_cache[key]

    def short_context(self, ticker: str) -> dict[str, Any]:
        return self.context_pack(ticker=ticker, query_mode="short", generate_hub_card=False)

    def long_context(self, ticker: str, *, generate_hub_card: bool) -> dict[str, Any]:
        return self.context_pack(ticker=ticker, query_mode="long", generate_hub_card=generate_hub_card)

    def context_pack(
        self,
        ticker: str,
        *,
        query_mode: str,
        generate_hub_card: bool,
    ) -> dict[str, Any]:
        ticker_norm = str(ticker or "").strip().upper()
        mode = str(query_mode or "").strip().lower()
        if mode not in {"short", "long"}:
            mode = "short"
        key = (ticker_norm, mode, bool(generate_hub_card))
        if key in self._context_cache:
            return self._context_cache[key]

        if not callable(self.context_loader):
            pack = _degraded_context_stub(ticker=ticker_norm, query_mode=mode)
            self._context_cache[key] = pack
            return pack

        if mode == "long":
            interval = self.long_interval
            lookback_days = int(self.long_lookback_days)
        else:
            interval = self.short_interval
            lookback_days = int(self.short_lookback_days)

        pack = self.context_loader(
            ticker=ticker_norm,
            generate_hub_card=bool(generate_hub_card),
            interval=interval,
            lookback_days=lookback_days,
        )
        if isinstance(pack, dict):
            meta = pack.setdefault("meta", {})
            if isinstance(meta, dict):
                meta.setdefault("query_mode", mode)
                meta.setdefault("query_contract", {
                    "interval": interval,
                    "lookback_days": lookback_days,
                    "generate_hub_card": bool(generate_hub_card),
                })
        self._context_cache[key] = pack
        return pack

    def quote_snapshot(self, ticker: str) -> QuoteSnapshot:
        ticker_norm = str(ticker or "").strip().upper()
        if ticker_norm not in self._quote_cache:
            self._quote_cache[ticker_norm] = get_quote_snapshot(
                ticker=ticker_norm,
                fetcher=self._fetcher,
                series_resolver=lambda rk: self.chart_series(ticker_norm, rk),
            )
        return self._quote_cache[ticker_norm]

    def quote_snapshot_dict(self, ticker: str) -> dict[str, Any]:
        return quote_snapshot_to_dict(self.quote_snapshot(ticker=ticker))

    def pulse_card_data(self, ticker: str) -> dict[str, Any]:
        ticker_norm = str(ticker or "").strip().upper()
        return {
            "ticker": ticker_norm,
            "context_pack": self.short_context(ticker_norm),
            "quote": self.quote_snapshot_dict(ticker_norm),
            "series_1d": self.chart_series(ticker=ticker_norm, range_key="1D"),
        }

    def brain_card_data(self, ticker: str, *, generate_hub_card: bool) -> dict[str, Any]:
        ticker_norm = str(ticker or "").strip().upper()
        return {
            "ticker": ticker_norm,
            "context_pack": self.long_context(ticker_norm, generate_hub_card=generate_hub_card),
            "quote": self.quote_snapshot_dict(ticker_norm),
            "series_1d": self.chart_series(ticker=ticker_norm, range_key="1D"),
        }

    def horizon_card_data(self, ticker: str) -> dict[str, Any]:
        # Horizon follows short-query contract for speed.
        return self.pulse_card_data(ticker=ticker)

    def clear_local_cache(self) -> None:
        self._series_cache.clear()
        self._quote_cache.clear()
        self._context_cache.clear()

    def revalidate_tickers(self, *, tickers: set[str], range_keys: tuple[str, ...] = ("1D", "1W")) -> dict[str, int]:
        stats = {"attempted": 0, "live": 0, "cache": 0, "none": 0, "errors": 0}
        if not tickers:
            return stats

        normalized_ranges = tuple(str(key or "").strip().upper() for key in range_keys if str(key or "").strip())
        if not normalized_ranges:
            normalized_ranges = ("1D",)

        for ticker in sorted(str(symbol or "").strip().upper() for symbol in tickers if str(symbol or "").strip()):
            for range_key in normalized_ranges:
                stats["attempted"] += 1
                try:
                    series = self._fetcher.fetch_chart_series(
                        ticker=ticker,
                        range_key=range_key,
                        force_revalidate=True,
                    )
                except Exception:
                    stats["errors"] += 1
                    continue
                self._series_cache[(ticker, range_key)] = series
                source = str(series.source).strip().lower()
                if source == "live":
                    stats["live"] += 1
                elif source == "cache":
                    stats["cache"] += 1
                else:
                    stats["none"] += 1
        return stats


def _degraded_context_stub(ticker: str, query_mode: str) -> dict[str, Any]:
    return {
        "meta": {
            "ticker": str(ticker or "").strip().upper(),
            "query_mode": str(query_mode or "short"),
            "data_quality": {
                "notes": ["DEGRADED_UI", "QUERY_LOADER_UNAVAILABLE"],
            },
        },
        "prices": {"bars": []},
        "indicators": {"metrics": {}, "as_of": None},
        "drl": {"result": {"action_final": "WAIT", "confidence_cap": 0.0, "conflicts": [], "gates_triggered": []}},
        "hub_card": None,
    }
