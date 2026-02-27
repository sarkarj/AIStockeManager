from __future__ import annotations

from datetime import datetime
from typing import Any

from app.ui.components.trust_badges import compute_trust_state
from app.ui.components.ui_utils import action_to_pill_class, action_to_ui_label, safe_get

_UNSET = object()

def build_brain_view_model(
    context_pack: dict,
    *,
    quote: dict[str, Any] | None = None,
    primary_series_close: float | None | object = _UNSET,
    fallback_series_close: float | None | object = _UNSET,
    series_for_selected_range: dict[str, Any] | None = None,
) -> dict:
    drl_result = safe_get(context_pack, "drl.result", {}) or {}
    action_raw = _normalize_action(str(drl_result.get("action_final", "WAIT")))
    confidence_cap = _to_number(drl_result.get("confidence_cap", 0.0))

    bars = safe_get(context_pack, "prices.bars", []) or []
    default_close = _last_close_or_none(bars)
    indicators = safe_get(context_pack, "indicators.metrics", {}) or {}
    selected_series = series_for_selected_range or {}
    if not selected_series and primary_series_close is not _UNSET:
        # Backward-compatible path for tests/callers that pass only scalar close.
        fallback_ts = safe_get(context_pack, "prices.as_of")
        selected_series = {
            "bars": (
                []
                if _to_optional_number(primary_series_close) is None
                else [{"ts": str(fallback_ts or ""), "close": float(_to_optional_number(primary_series_close) or 0.0)}]
            )
        }
    display = compute_display_price(indicators=indicators, series_for_selected_range=selected_series)
    quote_info = _quote_to_dict(quote)
    quote_latest = _quote_latest_price(quote_info)
    if isinstance(quote_latest, (int, float)):
        display = {
            "display_price": float(quote_latest),
            "price_source": "quote_latest",
            "price_ts": _to_datetime_or_none(quote_info.get("latest_ts")),
            "price_sanity_flags": set(str(flag) for flag in quote_info.get("quality_flags", [])),
        }
    elif quote is not None:
        display = {
            "display_price": None,
            "price_source": "none",
            "price_ts": None,
            "price_sanity_flags": set(str(flag) for flag in quote_info.get("quality_flags", [])) | {"MISSING_QUOTE"},
        }
    else:
        if display["price_source"] == "none" and default_close is not None:
            display = {
                **display,
                "display_price": default_close,
                "price_source": "bars_close_last",
                "price_ts": _to_datetime_or_none(safe_get(context_pack, "prices.as_of")),
                "price_sanity_flags": set(display["price_sanity_flags"]),
            }
        if fallback_series_close is not _UNSET and _to_optional_number(fallback_series_close) is not None:
            if display["price_source"] == "none":
                display = {
                    **display,
                    "display_price": float(_to_optional_number(fallback_series_close) or 0.0),
                    "price_source": "bars_close_last",
                    "price_sanity_flags": set(display["price_sanity_flags"]) | {"MISSING_BARS"},
                }

    bars_last_close, _ = _extract_last_bar_close_and_ts(selected_series)
    if (
        isinstance(quote_latest, (int, float))
        and isinstance(bars_last_close, (int, float))
        and quote_latest > 0
    ):
        mismatch = abs(float(quote_latest) - float(bars_last_close))
        threshold = max(0.50, 0.003 * float(quote_latest))
        if mismatch > threshold:
            display["price_sanity_flags"] = set(display["price_sanity_flags"]) | {"PRICE_MISMATCH"}
            quote_info["note"] = (
                f"Price mismatch: quote={float(quote_latest):.4f}, "
                f"bars_last={float(bars_last_close):.4f}"
            )

    last_price = display["display_price"]

    hub_card = context_pack.get("hub_card")
    if isinstance(hub_card, dict):
        hub_meta = hub_card.get("meta", {})
        hub_summary = hub_card.get("summary", {})
        hub_mode = _normalize_hub_mode(hub_meta.get("mode"))
        one_liner = _to_optional_str(hub_summary.get("one_liner"))
        drivers = _to_list_of_dicts(hub_card.get("drivers", []))
        conflicts = _to_list_of_dicts(hub_card.get("conflicts", []))
        watch = _to_list_of_dicts(hub_card.get("watch", []))
    else:
        hub_mode = None
        one_liner = None
        drivers = []
        conflicts = []
        watch = []

    ticker = (
        _to_optional_str(safe_get(context_pack, "meta.ticker"))
        or _to_optional_str(safe_get(context_pack, "drl.result.decision_trace.ticker"))
        or ""
    )
    as_of = _to_optional_str(safe_get(context_pack, "prices.as_of"))

    return {
        "ticker": ticker,
        "last_price": last_price,
        "ui_action_label": action_to_ui_label(action_raw),
        "ui_action_pill_class": action_to_pill_class(action_raw),
        "drl_action_raw": action_raw,
        "confidence_cap": confidence_cap,
        "hub_mode": hub_mode,
        "one_liner": one_liner,
        "drivers": drivers,
        "conflicts": conflicts,
        "watch": watch,
        "price_sanity": {
            "source": str(display["price_source"]),
            "quality_flags": sorted(str(x) for x in display["price_sanity_flags"]),
            "note": quote_info.get("note") or _build_price_sanity_note(display=display, indicators=indicators),
            "indicator_price_last": _to_optional_number(indicators.get("price_last")),
            "display_price": display["display_price"],
            "price_ts": display["price_ts"].isoformat() if isinstance(display["price_ts"], datetime) else None,
        },
        "quote": quote_info,
        "badges": compute_trust_state(context_pack),
        "as_of": as_of,
    }


def compute_display_price(indicators: dict[str, Any], series_for_selected_range: dict[str, Any]) -> dict[str, Any]:
    indicator_price_last = _to_optional_number((indicators or {}).get("price_last"))
    last_bar_close, last_bar_ts = _extract_last_bar_close_and_ts(series_for_selected_range)
    flags: set[str] = set()

    if last_bar_close is not None:
        display_price = last_bar_close
        price_source = "bars_close_last"
        price_ts = last_bar_ts
    elif indicator_price_last is not None:
        display_price = indicator_price_last
        price_source = "indicator_price_last"
        price_ts = None
        flags.add("MISSING_BARS")
    else:
        display_price = None
        price_source = "none"
        price_ts = None
        flags.add("MISSING_BARS")

    if (
        last_bar_close is not None
        and indicator_price_last is not None
        and indicator_price_last != 0.0
    ):
        abs_pct_diff = abs(last_bar_close - indicator_price_last) / abs(indicator_price_last)
        if abs_pct_diff > 0.01:
            flags.add("PRICE_MISMATCH_GT_1PCT")

    return {
        "display_price": display_price,
        "price_source": price_source,
        "price_ts": price_ts,
        "price_sanity_flags": flags,
    }


def _normalize_action(value: str) -> str:
    if value in {"ACCUMULATE", "WAIT", "REDUCE"}:
        return value
    return "WAIT"


def _normalize_hub_mode(value: Any) -> str | None:
    mode = str(value) if value is not None else ""
    if mode in {"FULL", "TECHNICAL_ONLY", "DEGRADED"}:
        return mode
    return None


def _to_list_of_dicts(value: Any) -> list:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


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


def _extract_last_bar_close_and_ts(series_for_selected_range: dict[str, Any]) -> tuple[float | None, datetime | None]:
    bars = series_for_selected_range.get("bars", []) if isinstance(series_for_selected_range, dict) else []
    if not isinstance(bars, list) or not bars:
        return None, None
    last = bars[-1]
    if not isinstance(last, dict):
        return None, None
    close = _to_optional_number(last.get("close"))
    ts = _to_datetime_or_none(last.get("ts"))
    return close, ts


def _to_datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt


def _build_price_sanity_note(display: dict[str, Any], indicators: dict[str, Any]) -> str | None:
    indicator_price_last = _to_optional_number((indicators or {}).get("price_last"))
    display_price = _to_optional_number(display.get("display_price"))
    flags = {str(x) for x in (display.get("price_sanity_flags") or set())}
    if "PRICE_MISMATCH_GT_1PCT" in flags and indicator_price_last is not None and display_price is not None:
        return (
            f"display_price {display_price:.4f} differs from indicator_price_last "
            f"{indicator_price_last:.4f} by >1%"
        )
    if "MISSING_BARS" in flags and display.get("price_source") != "bars_close_last":
        return "selected range bars missing; fell back from bars source"
    return None


def _quote_to_dict(quote: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(quote, dict):
        after_price = _to_optional_number(quote.get("after_hours_price"))
        close_price = _to_optional_number(quote.get("close_price"))
        latest_ts = quote.get("after_hours_ts") if after_price is not None else quote.get("close_ts")
        return {
            "symbol": _to_optional_str(quote.get("symbol")),
            "currency": _to_optional_str(quote.get("currency")) or "USD",
            "close_price": close_price,
            "close_ts": str(quote.get("close_ts")) if quote.get("close_ts") else None,
            "close_ts_local": _to_optional_str(quote.get("close_ts_local")),
            "prev_close_price": _to_optional_number(quote.get("prev_close_price")),
            "last_regular": _to_optional_number(quote.get("last_regular")),
            "last_regular_ts": _to_optional_str(quote.get("last_regular_ts")),
            "after_hours_price": after_price,
            "after_hours_ts": str(quote.get("after_hours_ts")) if quote.get("after_hours_ts") else None,
            "after_hours_ts_local": _to_optional_str(quote.get("after_hours_ts_local")),
            "latest_price": _quote_latest_price(quote),
            "latest_ts": str(latest_ts) if latest_ts else None,
            "latest_ts_local": _to_optional_str(quote.get("latest_ts_local")),
            "latest_source": _to_optional_str(quote.get("latest_source")) or ("after_hours" if after_price is not None else "close"),
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


def _quote_latest_price(value: dict[str, Any] | None) -> float | None:
    if isinstance(value, dict):
        latest = _to_optional_number(value.get("latest_price"))
        if latest is not None:
            return latest
        after = _to_optional_number(value.get("after_hours_price"))
        close = _to_optional_number(value.get("close_price"))
        return after if after is not None else close
    return None
