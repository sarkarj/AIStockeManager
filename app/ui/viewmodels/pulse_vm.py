from __future__ import annotations

from typing import Any

from app.core.marketdata.price_sanity import reconcile_price_last
from app.ui.components.ui_utils import action_to_pill_class, action_to_ui_label, safe_get

_UNSET = object()

def build_pulse_row_vm(
    holding: dict,
    context_pack: dict,
    *,
    quote: dict[str, Any] | None = None,
    primary_series_close: float | None | object = _UNSET,
    fallback_series_close: float | None | object = _UNSET,
) -> dict:
    ticker = _holding_value(holding, "ticker")
    qty = _to_number(_holding_value(holding, "quantity"))
    avg_cost = _to_number(_holding_value(holding, "avg_cost"))
    default_close = _last_close_or_none(safe_get(context_pack, "prices.bars", []) or [])
    indicator_price_last = _to_optional_number(safe_get(context_pack, "indicators.metrics.price_last"))
    primary_close_value = default_close if primary_series_close is _UNSET else _to_optional_number(primary_series_close)
    fallback_close_value = None if fallback_series_close is _UNSET else _to_optional_number(fallback_series_close)
    reconcile_sanity = reconcile_price_last(
        ticker=str(ticker or ""),
        indicator_price_last=indicator_price_last,
        primary_series_close=primary_close_value,
        fallback_series_close=fallback_close_value,
    )

    quote_info = _quote_to_dict(quote)
    quote_latest = _quote_latest_price(quote_info)
    sanity_flags = {str(flag) for flag in quote_info["quality_flags"]}
    sanity_note = quote_info["note"]
    if quote_latest is None and quote is None:
        quote_latest = reconcile_sanity.display_price
        sanity_flags.update(reconcile_sanity.quality_flags)
        sanity_note = reconcile_sanity.note
    elif quote_latest is None:
        sanity_flags.add("MISSING_QUOTE")

    if (
        isinstance(quote_latest, (int, float))
        and isinstance(primary_close_value, (int, float))
        and quote_latest > 0
    ):
        mismatch = abs(float(quote_latest) - float(primary_close_value))
        threshold = max(0.50, 0.003 * float(quote_latest))
        if mismatch > threshold:
            sanity_flags.add("PRICE_MISMATCH")
            sanity_note = (
                f"Price mismatch: quote={float(quote_latest):.4f}, "
                f"bars_last={float(primary_close_value):.4f}"
            )

    last_price = float(quote_latest) if isinstance(quote_latest, (int, float)) else None

    pl_dollars = None
    pl_pct = None
    if last_price is not None and qty is not None and avg_cost is not None:
        pl_dollars = (last_price - avg_cost) * qty
        pl_pct = ((last_price - avg_cost) / avg_cost * 100.0) if avg_cost > 0 else 0.0

    drl_result = safe_get(context_pack, "drl.result", {}) or {}
    action_raw = _normalize_action(str(drl_result.get("action_final", "WAIT")))
    confidence_cap = _to_number(drl_result.get("confidence_cap", 0.0))
    freshness = safe_get(context_pack, "meta.data_quality.prices", {}) or {}
    market_value = None
    if last_price is not None:
        market_value = qty * last_price

    today_abs = _to_optional_number(quote_info.get("today_change_abs"))
    today_pct = _to_optional_number(quote_info.get("today_change_pct"))

    return {
        "ticker": ticker,
        "last_price": last_price,
        "avg_cost": avg_cost,
        "quantity": qty,
        "market_value": market_value,
        "pl_dollars": pl_dollars,
        "pl_pct": pl_pct,
        "ui_action_label": action_to_ui_label(action_raw),
        "ui_action_pill_class": action_to_pill_class(action_raw),
        "drl_action_raw": action_raw,
        "confidence_cap": confidence_cap,
        "quote": quote_info,
        "today_abs": today_abs,
        "today_pct": today_pct,
        "freshness": freshness if isinstance(freshness, dict) else {},
        "price_sanity": {
            "source": quote_info["source"] if quote is not None else reconcile_sanity.source,
            "quality_flags": sorted(sanity_flags),
            "note": sanity_note,
            "indicator_price_last": indicator_price_last,
        },
        "badges": {},
    }


def _holding_value(holding: Any, key: str) -> Any:
    if isinstance(holding, dict):
        return holding.get(key)
    return getattr(holding, key, None)


def _normalize_action(value: str) -> str:
    if value in {"ACCUMULATE", "WAIT", "REDUCE"}:
        return value
    return "WAIT"


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_optional_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _last_close_or_none(bars: Any) -> float | None:
    if not isinstance(bars, list) or not bars:
        return None
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    try:
        return float(last.get("close"))
    except (TypeError, ValueError):
        return None


def _quote_to_dict(quote: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(quote, dict):
        return {
            "symbol": _to_optional_str(quote.get("symbol")),
            "currency": _to_optional_str(quote.get("currency")) or "USD",
            "close_price": _to_optional_number(quote.get("close_price")),
            "close_ts": str(quote.get("close_ts")) if quote.get("close_ts") else None,
            "close_ts_local": _to_optional_str(quote.get("close_ts_local")),
            "prev_close_price": _to_optional_number(quote.get("prev_close_price")),
            "last_regular": _to_optional_number(quote.get("last_regular")),
            "last_regular_ts": _to_optional_str(quote.get("last_regular_ts")),
            "after_hours_price": _to_optional_number(quote.get("after_hours_price")),
            "after_hours_ts": str(quote.get("after_hours_ts")) if quote.get("after_hours_ts") else None,
            "after_hours_ts_local": _to_optional_str(quote.get("after_hours_ts_local")),
            "latest_price": _to_optional_number(quote.get("latest_price")),
            "latest_ts": str(quote.get("latest_ts")) if quote.get("latest_ts") else None,
            "latest_ts_local": _to_optional_str(quote.get("latest_ts_local")),
            "latest_source": _to_optional_str(quote.get("latest_source")),
            "today_change_abs": _to_optional_number(quote.get("today_change_abs")),
            "today_change_pct": _to_optional_number(quote.get("today_change_pct")),
            "after_hours_change_abs": _to_optional_number(quote.get("after_hours_change_abs")),
            "after_hours_change_pct": _to_optional_number(quote.get("after_hours_change_pct")),
            "session_state": _to_optional_str(quote.get("session_state")),
            "show_extended_session": bool(quote.get("show_extended_session", False)),
            "extended_label": _to_optional_str(quote.get("extended_label")),
            "source": str(quote.get("source", "none")),
            "quality_flags": [str(flag) for flag in (quote.get("quality_flags", []) or [])],
            "error": str(quote.get("error")) if quote.get("error") else None,
            "note": str(quote.get("note")) if quote.get("note") else None,
        }
    return {
        "symbol": None,
        "currency": "USD",
        "close_price": None,
        "close_ts": None,
        "close_ts_local": None,
        "prev_close_price": None,
        "last_regular": None,
        "last_regular_ts": None,
        "after_hours_price": None,
        "after_hours_ts": None,
        "after_hours_ts_local": None,
        "latest_price": None,
        "latest_ts": None,
        "latest_ts_local": None,
        "latest_source": "none",
        "today_change_abs": None,
        "today_change_pct": None,
        "after_hours_change_abs": None,
        "after_hours_change_pct": None,
        "session_state": None,
        "show_extended_session": False,
        "extended_label": None,
        "source": "none",
        "quality_flags": [],
        "error": None,
        "note": None,
    }


def _quote_latest_price(quote_info: dict[str, Any]) -> float | None:
    latest = _to_optional_number(quote_info.get("latest_price"))
    if latest is not None:
        return latest
    after = _to_optional_number(quote_info.get("after_hours_price"))
    close = _to_optional_number(quote_info.get("close_price"))
    return after if after is not None else close


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
