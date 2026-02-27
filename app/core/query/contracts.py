from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from app.core.context_pack.why_cache import build_why_signature
from app.core.marketdata.query_graph import MarketQueryService
from app.core.orchestration.time_utils import now_iso, parse_iso

RangeKey = Literal["1D", "1W", "1M", "3M", "YTD", "1Y", "Advanced"]
_RANGES = {"1D", "1W", "1M", "3M", "YTD", "1Y"}


@dataclass(frozen=True)
class ShortQueryResult:
    ticker: str
    quote: dict[str, Any]
    series_1d: Any
    series_1w: Any
    drl_result: dict[str, Any]
    context_pack: dict[str, Any]
    as_of: datetime


@dataclass(frozen=True)
class LongQueryResult:
    ticker: str
    range_key: RangeKey
    quote: dict[str, Any]
    selected_series: Any
    drl_result: dict[str, Any]
    context_pack: dict[str, Any]
    why_signature: str | None
    as_of: datetime


def run_short_query(
    ticker: str,
    *,
    market_query: MarketQueryService,
    now: datetime | None = None,
) -> ShortQueryResult:
    symbol = str(ticker or "").strip().upper()
    payload = market_query.pulse_card_data(ticker=symbol)
    context_pack = payload.get("context_pack", {}) if isinstance(payload, dict) else {}
    if not isinstance(context_pack, dict):
        context_pack = {}
    quote = payload.get("quote", {}) if isinstance(payload, dict) else {}
    if not isinstance(quote, dict):
        quote = {}
    series_1d = payload.get("series_1d") if isinstance(payload, dict) else None
    chart_series_fn = getattr(market_query, "chart_series", None)
    if callable(chart_series_fn):
        series_1w = chart_series_fn(ticker=symbol, range_key="1W")
    else:
        series_1w = None
    drl_result = _drl_result_from_context(context_pack)
    as_of = now or parse_iso(now_iso())
    return ShortQueryResult(
        ticker=symbol,
        quote=quote,
        series_1d=series_1d,
        series_1w=series_1w,
        drl_result=drl_result,
        context_pack=context_pack,
        as_of=as_of,
    )


def run_long_query(
    ticker: str,
    *,
    range_key: RangeKey,
    include_why: bool,
    market_query: MarketQueryService,
    now: datetime | None = None,
) -> LongQueryResult:
    symbol = str(ticker or "").strip().upper()
    normalized_range = _normalize_range_key(range_key)
    payload = market_query.brain_card_data(ticker=symbol, generate_hub_card=bool(include_why))
    context_pack = payload.get("context_pack", {}) if isinstance(payload, dict) else {}
    if not isinstance(context_pack, dict):
        context_pack = {}
    quote = payload.get("quote", {}) if isinstance(payload, dict) else {}
    if not isinstance(quote, dict):
        quote = {}
    chart_series_fn = getattr(market_query, "chart_series", None)
    if callable(chart_series_fn):
        selected_series = chart_series_fn(ticker=symbol, range_key=normalized_range)
    else:
        selected_series = payload.get("series_1d") if isinstance(payload, dict) else None
    drl_result = _drl_result_from_context(context_pack)
    why_signature = build_why_signature(
        ticker=symbol,
        drl_result=drl_result,
        indicators=(context_pack.get("indicators", {}) if isinstance(context_pack, dict) else {}),
        quote=quote,
        range_key=normalized_range,
    )
    as_of = now or parse_iso(now_iso())
    return LongQueryResult(
        ticker=symbol,
        range_key=range_key,
        quote=quote,
        selected_series=selected_series,
        drl_result=drl_result,
        context_pack=context_pack,
        why_signature=why_signature,
        as_of=as_of,
    )


def _drl_result_from_context(context_pack: dict[str, Any]) -> dict[str, Any]:
    drl = context_pack.get("drl", {}) if isinstance(context_pack, dict) else {}
    if not isinstance(drl, dict):
        return {}
    result = drl.get("result", {})
    return result if isinstance(result, dict) else {}


def _normalize_range_key(range_key: str) -> str:
    key = str(range_key or "1D").strip().upper()
    if key == "ADVANCED":
        return "3M"
    if key not in _RANGES:
        return "1D"
    return key
