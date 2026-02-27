from __future__ import annotations

import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.core.marketdata.query_graph import MarketQueryService
from app.core.query.contracts import run_long_query, run_short_query
from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import now_iso

_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
_POLICY_PATH = Path(__file__).resolve().parents[1] / "core" / "drl" / "policies" / "drl_policy.yaml"
_DEFAULT_LONG_RANGES = ("1D", "1W", "1M", "3M", "YTD", "1Y")
_DEFAULT_SHORT_RANGES = ("1D", "1W")
_REVALIDATE_LAST: dict[str, float] = {}


def create_app(
    service_factory: Callable[[], MarketQueryService] | None = None,
) -> Starlette:
    query_factory = service_factory or _default_query_factory
    app = Starlette(debug=False, routes=[
        Route("/api/health", endpoint=_health, methods=["GET"]),
        Route("/api/query/short", endpoint=_query_short, methods=["POST"]),
        Route("/api/query/long", endpoint=_query_long, methods=["POST"]),
    ])
    app.state.query_factory = query_factory
    app.state.query_singleton = None
    return app

async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "service": "graph-api", "as_of": now_iso()})


async def _query_short(request: Request) -> JSONResponse:
    auth_error = _check_auth(request)
    if auth_error:
        return auth_error

    payload, error = await _read_payload(request)
    if error:
        return error

    ticker = _normalize_ticker(payload.get("ticker"))
    if not ticker:
        return _error(422, "INVALID_TICKER", "Ticker must contain A-Z, 0-9, '.' or '-'.")

    revalidate = _to_bool(payload.get("revalidate", False))
    query = _get_query_service(request)
    if revalidate:
        ok, retry_after = _revalidate_allowed(endpoint="short", ticker=ticker)
        if not ok:
            return _error(
                429,
                "REVALIDATE_THROTTLED",
                f"Revalidate throttled; retry after {retry_after:.1f}s.",
            )
        query.revalidate_tickers(tickers={ticker}, range_keys=_DEFAULT_SHORT_RANGES)

    short_result = run_short_query(ticker=ticker, market_query=query)
    data = {
        "ticker": short_result.ticker,
        "context_pack": short_result.context_pack,
        "quote": short_result.quote,
        "series_1d": short_result.series_1d,
        "series_1w": short_result.series_1w,
        "drl_result": short_result.drl_result,
        "as_of": short_result.as_of,
    }
    response = {
        "ok": True,
        "query": "short",
        "ticker": ticker,
        "revalidated": bool(revalidate),
        "generated_at": now_iso(),
        "data": _serialize_query_payload(data),
    }
    return JSONResponse(response)


async def _query_long(request: Request) -> JSONResponse:
    auth_error = _check_auth(request)
    if auth_error:
        return auth_error

    payload, error = await _read_payload(request)
    if error:
        return error

    ticker = _normalize_ticker(payload.get("ticker"))
    if not ticker:
        return _error(422, "INVALID_TICKER", "Ticker must contain A-Z, 0-9, '.' or '-'.")

    generate_hub_card = _to_bool(payload.get("generate_hub_card", False))
    range_key = str(payload.get("range_key", "1D") or "1D")
    revalidate = _to_bool(payload.get("revalidate", False))
    query = _get_query_service(request)
    if revalidate:
        ok, retry_after = _revalidate_allowed(endpoint="long", ticker=ticker)
        if not ok:
            return _error(
                429,
                "REVALIDATE_THROTTLED",
                f"Revalidate throttled; retry after {retry_after:.1f}s.",
            )
        query.revalidate_tickers(tickers={ticker}, range_keys=_DEFAULT_LONG_RANGES)

    normalized_range = str(range_key).strip().upper() or "1D"
    if normalized_range not in {"1D", "1W", "1M", "3M", "YTD", "1Y", "ADVANCED"}:
        normalized_range = "1D"
    long_result = run_long_query(
        ticker=ticker,
        range_key=normalized_range,  # type: ignore[arg-type]
        include_why=generate_hub_card,
        market_query=query,
    )
    data = {
        "ticker": long_result.ticker,
        "range_key": long_result.range_key,
        "context_pack": long_result.context_pack,
        "quote": long_result.quote,
        "selected_series": long_result.selected_series,
        "drl_result": long_result.drl_result,
        "why_signature": long_result.why_signature,
        "as_of": long_result.as_of,
    }
    response = {
        "ok": True,
        "query": "long",
        "ticker": ticker,
        "generate_hub_card": bool(generate_hub_card),
        "revalidated": bool(revalidate),
        "generated_at": now_iso(),
        "data": _serialize_query_payload(data),
    }
    return JSONResponse(response)


def _default_query_factory() -> MarketQueryService:
    return MarketQueryService(
        cache_dir=".cache/charts",
        context_loader=_context_loader,
        short_interval="1h",
        short_lookback_days=60,
        long_interval="1h",
        long_lookback_days=60,
    )


def _context_loader(
    ticker: str,
    generate_hub_card: bool = False,
    interval: str = "1h",
    lookback_days: int = 60,
) -> dict[str, Any]:
    provider = SampleMarketDataProvider()
    cache = DiskTTLCache(base_dir=".cache")

    bedrock_region = os.getenv("AWS_REGION", "") or os.getenv("AWS_DEFAULT_REGION", "")
    bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "")
    bedrock_claude_model_id = os.getenv("BEDROCK_LLM_ID_CLAUDE", "")
    bedrock_openai_model_id = os.getenv("BEDROCK_LLM_ID_OPENAI", "")
    bedrock_config = None
    if bedrock_region.strip() and (
        bedrock_model_id.strip() or bedrock_claude_model_id.strip() or bedrock_openai_model_id.strip()
    ):
        bedrock_config = {
            "region": bedrock_region.strip(),
            "model_id": bedrock_model_id.strip(),
            "claude_model_id": bedrock_claude_model_id.strip(),
            "openai_model_id": bedrock_openai_model_id.strip(),
        }

    context_pack = build_context_pack(
        ticker=str(ticker).strip().upper(),
        now_iso=now_iso(),
        provider=provider,
        cache=cache,
        policy_path=str(_POLICY_PATH),
        lookback_days=int(lookback_days),
        interval=str(interval),
        generate_hub_card=bool(generate_hub_card),
        bedrock_config=bedrock_config,
    )
    return context_pack


def _get_query_service(request: Request) -> MarketQueryService:
    cached = getattr(request.app.state, "query_singleton", None)
    if cached is not None:
        required_cached = ("pulse_card_data", "brain_card_data", "revalidate_tickers")
        if all(callable(getattr(cached, name, None)) for name in required_cached):
            return cached

    query_factory = getattr(request.app.state, "query_factory", None)
    if not callable(query_factory):
        raise RuntimeError("query_factory_unavailable")
    query = query_factory()
    required = ("pulse_card_data", "brain_card_data", "revalidate_tickers")
    if not all(callable(getattr(query, name, None)) for name in required):
        raise RuntimeError("query_factory_invalid")
    request.app.state.query_singleton = query
    return query


async def _read_payload(request: Request) -> tuple[dict[str, Any], JSONResponse | None]:
    try:
        payload = await request.json()
    except Exception:
        return {}, _error(400, "INVALID_JSON", "Request body must be valid JSON.")
    if not isinstance(payload, dict):
        return {}, _error(400, "INVALID_PAYLOAD", "Request body must be a JSON object.")
    if len(payload) > 8:
        return {}, _error(400, "PAYLOAD_TOO_LARGE", "Payload has too many fields.")
    return payload, None


def _check_auth(request: Request) -> JSONResponse | None:
    configured_key = str(os.getenv("GRAPH_API_KEY", "")).strip()
    allow_unauth_local = _to_bool(os.getenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "0"))
    if allow_unauth_local and _is_localhost_client(request):
        return None
    if not configured_key:
        return _error(503, "AUTH_NOT_CONFIGURED", "GRAPH_API_KEY is required.")
    header_key = str(request.headers.get("x-api-key", "")).strip()
    if header_key and secrets.compare_digest(header_key, configured_key):
        return None
    auth_header = str(request.headers.get("authorization", "")).strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token and secrets.compare_digest(token, configured_key):
            return None
    return _error(401, "UNAUTHORIZED", "Missing or invalid API key.")


def _normalize_ticker(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if not _TICKER_PATTERN.fullmatch(text):
        return None
    return text


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _serialize_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload or {})
    for key, value in list(result.items()):
        if not str(key).startswith("series_") and str(key) != "selected_series":
            continue
        series = value
        if series is not None and hasattr(series, "bars") and hasattr(series, "source"):
            bars = []
            for bar in list(getattr(series, "bars", []) or []):
                bars.append(
                    {
                        "ts": bar.ts.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": None if bar.volume is None else float(bar.volume),
                    }
                )
            result[key] = {
                "bars": bars,
                "as_of": getattr(series, "as_of").isoformat() if getattr(series, "as_of", None) else None,
                "source": str(getattr(series, "source", "none")),
                "error": getattr(series, "error", None),
                "quality_flags": sorted(str(flag) for flag in (getattr(series, "quality_flags", set()) or set())),
                "cache_hit": bool(getattr(series, "cache_hit", False)),
                "cache_age_minutes": getattr(series, "cache_age_minutes", None),
                "attempts": int(getattr(series, "attempts", 0)),
            }
    return _jsonable(result)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return [_jsonable(v) for v in sorted(value, key=lambda item: str(item))]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": {"code": str(code), "message": str(message)}},
        status_code=int(status),
    )


def _is_localhost_client(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _revalidate_allowed(*, endpoint: str, ticker: str) -> tuple[bool, float]:
    now_ts = time.time()
    key = f"{str(endpoint).strip().lower()}:{str(ticker).strip().upper()}"
    last_ts = _REVALIDATE_LAST.get(key)
    if last_ts is None:
        _REVALIDATE_LAST[key] = now_ts
        return True, 0.0
    try:
        cooldown = float(os.getenv("GRAPH_API_REVALIDATE_COOLDOWN_SECONDS", "15"))
    except ValueError:
        cooldown = 15.0
    cooldown = float(max(1.0, cooldown))
    elapsed = now_ts - float(last_ts)
    if elapsed >= cooldown:
        _REVALIDATE_LAST[key] = now_ts
        return True, 0.0
    return False, max(0.0, cooldown - elapsed)


app = create_app()
