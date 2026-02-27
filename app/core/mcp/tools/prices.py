from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import now_iso as default_now_iso

if TYPE_CHECKING:
    from app.core.marketdata.provider import MarketDataProvider

TOOL_NAME = "stock.get_prices"
TOOL_DESCRIPTION = "Return normalized OHLCV bars for a ticker."
ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
        "interval": {"type": "string", "description": "Bar interval, e.g. 1h, 1d", "default": "1h"},
        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 60},
    },
    "required": ["ticker"],
}


def get_prices(
    ticker: str,
    interval: str = "1h",
    lookback_days: int = 60,
    provider: "MarketDataProvider | None" = None,
    cache: DiskTTLCache | None = None,
    ttl_seconds: int = 3600,
    now_iso: str | None = None,
) -> dict[str, Any]:
    request_now = now_iso or default_now_iso()
    ticker_norm = ticker.strip().upper()
    cache_key = f"prices:{ticker_norm}:{interval}:{int(lookback_days)}"

    try:
        if cache is None:
            cache = DiskTTLCache(base_dir=".cache")
        if provider is None:
            provider = SampleMarketDataProvider()

        cached_payload = cache.get(cache_key, now_iso=request_now)
        if cached_payload is not None:
            return _build_response(
                ticker=ticker_norm,
                interval=interval,
                lookback_days=int(lookback_days),
                payload=cached_payload,
                provider_name=_provider_name(provider),
                cached=True,
                cache_key=cache_key,
            )

        fetched = provider.get_ohlcv(ticker=ticker_norm, interval=interval, lookback_days=int(lookback_days))
        payload = {
            "as_of": str(fetched["as_of"]),
            "bars": [_normalize_bar(bar) for bar in fetched.get("bars", [])],
        }
        cache.set(cache_key, payload, ttl_seconds=ttl_seconds)

        return _build_response(
            ticker=ticker_norm,
            interval=interval,
            lookback_days=int(lookback_days),
            payload=payload,
            provider_name=_provider_name(provider),
            cached=False,
            cache_key=cache_key,
        )
    except Exception as exc:
        return _error_response(
            message=f"Unable to fetch prices for {ticker_norm}.",
            details={"reason": str(exc), "ticker": ticker_norm, "interval": interval},
        )


def _normalize_bar(bar: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": str(bar["ts"]),
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": None if bar.get("volume") is None else float(bar["volume"]),
    }


def _provider_name(provider: object) -> str:
    name = provider.__class__.__name__.lower()
    if name.endswith("provider"):
        name = name[: -len("provider")]
    return name or "unknown"


def _build_response(
    ticker: str,
    interval: str,
    lookback_days: int,
    payload: dict[str, Any],
    provider_name: str,
    cached: bool,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of": str(payload["as_of"]),
        "interval": interval,
        "lookback_days": int(lookback_days),
        "bars": list(payload.get("bars", [])),
        "source": {
            "provider": provider_name,
            "cached": bool(cached),
            "cache_key": cache_key,
        },
    }


def _error_response(message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": {
            "code": "TOOL_DOWN",
            "message": message,
            "details": details,
        }
    }
