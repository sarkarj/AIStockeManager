from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import now_iso as default_now_iso

if TYPE_CHECKING:
    from app.core.marketdata.provider import MarketDataProvider

TOOL_NAME = "stock.get_macro_snapshot"
TOOL_DESCRIPTION = "Return a normalized macro snapshot using broad market proxies."
ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 30, "default": 1},
    },
}


def get_macro_snapshot(
    lookback_days: int = 1,
    provider: "MarketDataProvider | None" = None,
    cache: DiskTTLCache | None = None,
    ttl_seconds: int = 3600,
    now_iso: str | None = None,
) -> dict[str, Any]:
    request_now = now_iso or default_now_iso()
    cache_key = f"macro:{int(lookback_days)}"

    try:
        if cache is None:
            cache = DiskTTLCache(base_dir=".cache")
        if provider is None:
            provider = SampleMarketDataProvider()

        cached_payload = cache.get(cache_key, now_iso=request_now)
        if cached_payload is not None:
            return _build_response(
                payload=cached_payload,
                provider_name=_provider_name(provider),
                cached=True,
                cache_key=cache_key,
            )

        spy_payload = provider.get_ohlcv(ticker="SPY", interval="1d", lookback_days=max(2, int(lookback_days) + 1))
        spy_bars = list(spy_payload.get("bars", []))
        spy_change_pct = _daily_pct_change(spy_bars)

        vix_last: float | None = None
        try:
            vix_payload = provider.get_ohlcv(ticker="^VIX", interval="1d", lookback_days=max(2, int(lookback_days) + 1))
            vix_bars = list(vix_payload.get("bars", []))
            if vix_bars:
                vix_last = float(vix_bars[-1]["close"])
        except Exception:
            vix_last = None

        items: list[dict[str, Any]] = [
            {
                "id": "SPY_CHANGE_1D",
                "label": "SPY % Change (1D)",
                "value": round(spy_change_pct, 4),
                "source": "SPY",
            }
        ]
        if vix_last is not None:
            items.append(
                {
                    "id": "VIX_LAST",
                    "label": "VIX Last",
                    "value": round(vix_last, 4),
                    "source": "^VIX",
                }
            )

        market_mode = _compute_market_mode(spy_change_pct=spy_change_pct, vix_last=vix_last)

        payload = {
            "as_of": str(spy_payload.get("as_of", request_now)),
            "items": items,
            "market_mode": market_mode,
        }
        cache.set(cache_key, payload, ttl_seconds=ttl_seconds)

        return _build_response(
            payload=payload,
            provider_name=_provider_name(provider),
            cached=False,
            cache_key=cache_key,
        )
    except Exception as exc:
        return _error_response(
            message="Unable to compute macro snapshot.",
            details={"reason": str(exc), "lookback_days": int(lookback_days)},
        )


def _daily_pct_change(bars: list[dict[str, Any]]) -> float:
    if len(bars) < 2:
        return 0.0
    prev_close = float(bars[-2]["close"])
    last_close = float(bars[-1]["close"])
    if prev_close == 0:
        return 0.0
    return ((last_close - prev_close) / prev_close) * 100.0


def _compute_market_mode(spy_change_pct: float, vix_last: float | None) -> dict[str, Any]:
    if vix_last is not None:
        if spy_change_pct > 1.0 and vix_last < 20.0:
            return {"label": "Risk On", "confidence": 75}
        if spy_change_pct < -1.0 and vix_last > 20.0:
            return {"label": "Risk Off", "confidence": 75}
        return {"label": "Neutral", "confidence": 60}

    if spy_change_pct > 1.0:
        return {"label": "Risk On", "confidence": 50}
    if spy_change_pct < -1.0:
        return {"label": "Risk Off", "confidence": 50}
    return {"label": "Neutral", "confidence": 40}


def _provider_name(provider: object) -> str:
    name = provider.__class__.__name__.lower()
    if name.endswith("provider"):
        name = name[: -len("provider")]
    return name or "unknown"


def _build_response(payload: dict[str, Any], provider_name: str, cached: bool, cache_key: str) -> dict[str, Any]:
    return {
        "as_of": str(payload.get("as_of", "")),
        "items": list(payload.get("items", [])),
        "market_mode": dict(payload.get("market_mode", {"label": "Neutral", "confidence": 0})),
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
