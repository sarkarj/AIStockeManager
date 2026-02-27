from __future__ import annotations

import hashlib
from typing import Any

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import now_iso as default_now_iso

TOOL_NAME = "stock.get_events"
TOOL_DESCRIPTION = "Return normalized notable events for a ticker (v1 stub)."
ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "max_items": {"type": "integer", "minimum": 0, "maximum": 20, "default": 10},
    },
    "required": ["ticker"],
}


def get_events(
    ticker: str,
    max_items: int = 10,
    cache: DiskTTLCache | None = None,
    ttl_seconds: int = 3600,
    now_iso: str | None = None,
) -> dict[str, Any]:
    request_now = now_iso or default_now_iso()
    ticker_norm = ticker.strip().upper()
    cache_key = f"events:{ticker_norm}:{int(max_items)}"

    try:
        if cache is None:
            cache = DiskTTLCache(base_dir=".cache")

        cached_payload = cache.get(cache_key, now_iso=request_now)
        if cached_payload is not None:
            return _build_response(
                payload=cached_payload,
                ticker=ticker_norm,
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
            provider_name="stub",
            cached=False,
            cache_key=cache_key,
        )
    except Exception as exc:
        return _error_response(
            message=f"Unable to fetch events for {ticker_norm}.",
            details={"reason": str(exc), "ticker": ticker_norm},
        )


def stable_event_id(ticker: str, event_type: str, event_date: str, title: str) -> str:
    material = f"{ticker}:{event_type}:{event_date}:{title}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _build_response(
    payload: dict[str, Any],
    ticker: str,
    provider_name: str,
    cached: bool,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of": str(payload.get("as_of", "")),
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
