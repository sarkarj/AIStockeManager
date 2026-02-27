from __future__ import annotations

import hashlib
from typing import Any

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import now_iso as default_now_iso

TOOL_NAME = "stock.get_news"
TOOL_DESCRIPTION = "Return normalized recent news items for a ticker."
ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "lookback_hours": {"type": "integer", "minimum": 1, "maximum": 168, "default": 48},
        "max_items": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12},
    },
    "required": ["ticker"],
}


def get_news(
    ticker: str,
    lookback_hours: int = 48,
    max_items: int = 12,
    cache: DiskTTLCache | None = None,
    ttl_seconds: int = 3600,
    now_iso: str | None = None,
) -> dict[str, Any]:
    request_now = now_iso or default_now_iso()
    ticker_norm = ticker.strip().upper()
    cache_key = f"news:{ticker_norm}:{int(lookback_hours)}:{int(max_items)}"

    try:
        if cache is None:
            cache = DiskTTLCache(base_dir=".cache")

        cached_payload = cache.get(cache_key, now_iso=request_now)
        if cached_payload is not None:
            return _build_response(
                payload=cached_payload,
                ticker=ticker_norm,
                lookback_hours=int(lookback_hours),
                provider_name="stub",
                cached=True,
                cache_key=cache_key,
            )

        # Phase 3 v1 provider: deterministic empty set (read-only stub).
        payload = {
            "as_of": request_now,
            "items": [],
        }
        cache.set(cache_key, payload, ttl_seconds=ttl_seconds)

        return _build_response(
            payload=payload,
            ticker=ticker_norm,
            lookback_hours=int(lookback_hours),
            provider_name="stub",
            cached=False,
            cache_key=cache_key,
        )
    except Exception as exc:
        return _error_response(
            message=f"Unable to fetch news for {ticker_norm}.",
            details={"reason": str(exc), "ticker": ticker_norm},
        )


def stable_news_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def normalize_news_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    url = str(raw_item.get("url", ""))
    return {
        "id": stable_news_id(url),
        "source": str(raw_item.get("source", "unknown")),
        "published_at": str(raw_item.get("published_at", "")),
        "url": url,
        "title": str(raw_item.get("title", "")),
        "summary": str(raw_item.get("summary", "")),
    }


def _build_response(
    payload: dict[str, Any],
    ticker: str,
    lookback_hours: int,
    provider_name: str,
    cached: bool,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of": str(payload.get("as_of", "")),
        "lookback_hours": int(lookback_hours),
        "items": list(payload.get("items", [])),
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
