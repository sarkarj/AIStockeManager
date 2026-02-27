from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

import requests

from app.core.orchestration.time_utils import now_iso, parse_iso

_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


class YahooChartMarketDataProvider:
    """Free Yahoo chart endpoint provider for live OHLCV bars."""

    BASE_URLS = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    )

    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        ticker_norm = _normalize_ticker(ticker)
        if not ticker_norm:
            raise ValueError("Ticker is required.")

        interval_norm = _normalize_interval(interval)
        lookback = max(1, int(lookback_days))

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback)
        params = {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": interval_norm,
            "includePrePost": "false",
            "events": "div,splits",
        }

        payload = None
        last_error: Exception | None = None
        for base_url in self.BASE_URLS:
            try:
                response = requests.get(
                    base_url.format(ticker=ticker_norm),
                    params=params,
                    timeout=10,
                    headers=_YAHOO_HEADERS,
                )
                response.raise_for_status()
                payload = response.json() or {}
                break
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                continue
        if payload is None:
            raise RuntimeError(f"Yahoo provider unavailable for {ticker_norm}: {last_error}")

        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RuntimeError(f"Yahoo provider error for {ticker_norm}.")

        results = chart.get("result", [])
        if not results:
            raise RuntimeError(f"No Yahoo chart results for {ticker_norm}.")

        result = results[0]
        ts_values = result.get("timestamp", []) or []
        quote_items = ((result.get("indicators") or {}).get("quote") or [])
        if not ts_values or not quote_items:
            raise RuntimeError(f"No Yahoo quote bars for {ticker_norm}.")

        quote = quote_items[0]
        opens = quote.get("open", []) or []
        highs = quote.get("high", []) or []
        lows = quote.get("low", []) or []
        closes = quote.get("close", []) or []
        volumes = quote.get("volume", []) or []

        bars: list[dict] = []
        for i, ts in enumerate(ts_values):
            if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
                continue
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            if o is None or h is None or l is None or c is None:
                continue
            volume = volumes[i] if i < len(volumes) else None
            ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            bars.append(
                {
                    "ts": ts_iso,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": None if volume is None else float(volume),
                }
            )

        if not bars:
            raise RuntimeError(f"No usable Yahoo bars for {ticker_norm}.")

        return {"as_of": bars[-1]["ts"], "bars": bars}


class SampleMarketDataProvider:
    """Deterministic local sample provider used until a live adapter is introduced."""

    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        step = _interval_to_timedelta(interval)
        bars_count = _bars_count(interval=interval, lookback_days=lookback_days)

        as_of = now_iso()
        as_of_dt = parse_iso(as_of)
        start_dt = as_of_dt - step * (bars_count - 1)

        ticker_seed = sum(ord(ch) for ch in ticker.upper())
        base = 90.0 + float(ticker_seed % 120)
        drift = ((ticker_seed % 7) - 3) * 0.03

        bars: list[dict] = []
        prev_close = base
        for i in range(bars_count):
            ts_dt = start_dt + step * i
            wave = 1.5 * math.sin((i + (ticker_seed % 11)) / 8.0)
            close = max(1.0, base + drift * i + wave)
            open_px = prev_close
            spread = 0.6 + 0.2 * abs(math.sin(i / 6.0))
            high = max(open_px, close) + spread
            low = min(open_px, close) - spread
            volume = float(800_000 + (ticker_seed % 300) * 1000 + (i % 24) * 2500)

            bars.append(
                {
                    "ts": ts_dt.isoformat(),
                    "open": float(open_px),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": volume,
                }
            )
            prev_close = close

        return {"as_of": as_of, "bars": bars}


class LiveOrSampleMarketDataProvider:
    """Prefer live Yahoo bars, fallback to deterministic sample provider."""

    def __init__(self):
        self.live = YahooChartMarketDataProvider()
        self.sample = SampleMarketDataProvider()

    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        mode = str(os.getenv("MARKETDATA_MODE", "live")).strip().lower()
        if mode == "sample":
            return self.sample.get_ohlcv(ticker=ticker, interval=interval, lookback_days=lookback_days)
        try:
            return self.live.get_ohlcv(ticker=ticker, interval=interval, lookback_days=lookback_days)
        except Exception:
            return self.sample.get_ohlcv(ticker=ticker, interval=interval, lookback_days=lookback_days)


def _interval_to_timedelta(interval: str) -> timedelta:
    value = interval.strip().lower()
    if value.endswith("m"):
        return timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return timedelta(hours=int(value[:-1]))
    if value.endswith("d"):
        return timedelta(days=int(value[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def _bars_count(interval: str, lookback_days: int) -> int:
    value = interval.strip().lower()
    if value.endswith("m"):
        bars = lookback_days * 24 * 60 // int(value[:-1])
    elif value.endswith("h"):
        bars = lookback_days * 24 // int(value[:-1])
    elif value.endswith("d"):
        bars = lookback_days // int(value[:-1])
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    return max(30, int(bars))


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _normalize_interval(interval: str) -> str:
    value = interval.strip().lower()
    supported = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
    if value in supported:
        return value
    # Fallback to 1h for unsupported UI requests.
    return "1h"
