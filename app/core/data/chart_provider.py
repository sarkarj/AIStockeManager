from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import now_iso

_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


class YahooChartProvider:
    BASE_URLS = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    )

    def __init__(
        self,
        cache: DiskTTLCache | None = None,
        cache_dir: str = ".cache",
        default_ttl_seconds: int = 1800,
        timeout_seconds: int = 8,
        max_retries: int = 2,
        backoff_seconds: float = 0.35,
    ) -> None:
        self.cache = cache or DiskTTLCache(base_dir=cache_dir)
        self.default_ttl_seconds = int(default_ttl_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

    def fetch_chart_series(
        self,
        ticker: str,
        period: str,
        interval: str,
        prepost: bool,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        ticker_norm = str(ticker).strip().upper()
        period_norm = str(period).strip()
        interval_norm = str(interval).strip().lower()
        prepost_norm = bool(prepost)

        ttl = int(ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds)
        request_now = now_iso()
        cache_key = f"chart:{ticker_norm}:{period_norm}:{interval_norm}:{int(prepost_norm)}"

        cached_payload = self.cache.get(cache_key, now_iso=request_now)
        if cached_payload is not None:
            return self._build_response(
                payload=cached_payload,
                provider_name="yahoo_chart",
                cache_key=cache_key,
                cached=True,
                stale_cache=False,
                request_now=request_now,
                attempts=0,
                errors=[],
            )

        attempts = 0
        errors: list[str] = []
        for retry in range(self.max_retries + 1):
            attempts += 1
            try:
                payload = self._fetch_remote(
                    ticker=ticker_norm,
                    period=period_norm,
                    interval=interval_norm,
                    prepost=prepost_norm,
                )
                self.cache.set(cache_key, payload, ttl_seconds=ttl)
                return self._build_response(
                    payload=payload,
                    provider_name="yahoo_chart",
                    cache_key=cache_key,
                    cached=False,
                    stale_cache=False,
                    request_now=request_now,
                    attempts=attempts,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                if retry < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**retry))

        stale_payload = self._read_stale_payload(cache_key)
        if stale_payload is not None:
            return self._build_response(
                payload=stale_payload,
                provider_name="yahoo_chart",
                cache_key=cache_key,
                cached=True,
                stale_cache=True,
                request_now=request_now,
                attempts=attempts,
                errors=errors,
            )

        return self._build_response(
            payload={"as_of": request_now, "bars": []},
            provider_name="yahoo_chart",
            cache_key=cache_key,
            cached=False,
            stale_cache=False,
            request_now=request_now,
            attempts=attempts,
            errors=errors,
        )

    def _fetch_remote(self, ticker: str, period: str, interval: str, prepost: bool) -> dict[str, Any]:
        params = {
            "range": period,
            "interval": interval,
            "includePrePost": "true" if prepost else "false",
            "events": "div,splits",
        }
        payload = None
        last_error: Exception | None = None
        for base_url in self.BASE_URLS:
            try:
                response = requests.get(
                    base_url.format(ticker=ticker),
                    params=params,
                    timeout=self.timeout_seconds,
                    headers=_YAHOO_HEADERS,
                )
                response.raise_for_status()
                payload = response.json() or {}
                break
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                continue
        if payload is None:
            raise RuntimeError(f"Yahoo chart unavailable for {ticker}: {last_error}")

        chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
        chart_error = chart.get("error")
        if chart_error:
            raise RuntimeError(f"Yahoo chart error: {chart_error}")

        results = chart.get("result", [])
        if not isinstance(results, list) or not results:
            raise RuntimeError("Yahoo chart result missing")

        result = results[0] if isinstance(results[0], dict) else {}
        timestamps = result.get("timestamp", [])
        quote_items = ((result.get("indicators") or {}).get("quote") or [])
        quote = quote_items[0] if isinstance(quote_items, list) and quote_items else {}

        opens = quote.get("open", []) if isinstance(quote, dict) else []
        highs = quote.get("high", []) if isinstance(quote, dict) else []
        lows = quote.get("low", []) if isinstance(quote, dict) else []
        closes = quote.get("close", []) if isinstance(quote, dict) else []
        volumes = quote.get("volume", []) if isinstance(quote, dict) else []

        bars: list[dict[str, Any]] = []
        for idx, ts in enumerate(timestamps or []):
            if idx >= len(opens) or idx >= len(highs) or idx >= len(lows) or idx >= len(closes):
                continue
            o = opens[idx]
            h = highs[idx]
            l = lows[idx]
            c = closes[idx]
            if o is None or h is None or l is None or c is None:
                continue
            vol = volumes[idx] if idx < len(volumes) else None
            ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            bars.append(
                {
                    "ts": ts_iso,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": None if vol is None else float(vol),
                }
            )

        bars.sort(key=lambda item: item["ts"])
        if not bars:
            raise RuntimeError("Yahoo chart returned no usable OHLC bars")

        return {"as_of": bars[-1]["ts"], "bars": bars}

    def _read_stale_payload(self, cache_key: str) -> dict[str, Any] | None:
        path = self.cache.path_for_key(cache_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        payload = record.get("payload")
        if isinstance(payload, dict):
            return payload
        return None

    def _build_response(
        self,
        payload: dict[str, Any],
        provider_name: str,
        cache_key: str,
        cached: bool,
        stale_cache: bool,
        request_now: str,
        attempts: int,
        errors: list[str],
    ) -> dict[str, Any]:
        bars = payload.get("bars", []) if isinstance(payload, dict) else []
        as_of = payload.get("as_of") if isinstance(payload, dict) else None
        return {
            "ticker": None,
            "as_of": str(as_of or request_now),
            "bars": bars if isinstance(bars, list) else [],
            "source": {
                "provider": provider_name,
                "cached": bool(cached),
                "stale_cache": bool(stale_cache),
                "cache_key": cache_key,
                "attempts": int(attempts),
            },
            "errors": [str(x) for x in errors],
        }
