from __future__ import annotations

import math
from typing import Any

from app.core.orchestration.time_utils import minutes_between

TOOLTIP_AGE = "Age = how old the price snapshot is (minutes since prices were fetched)."
TOOLTIP_DEGRADED = "Degraded = data quality issue (stale/missing/insufficient/suspect series). Signal shown is still DRL; confidence may be capped."
TOOLTIP_TOOL_DOWN = "Tool Down = price data fetch failed. Showing last known cached values if available."


def compute_pulse_badges(context_pack: dict) -> dict:
    meta = context_pack.get("meta", {}) if isinstance(context_pack, dict) else {}
    data_quality = meta.get("data_quality", {}) if isinstance(meta, dict) else {}
    prices_quality = data_quality.get("prices", {}) if isinstance(data_quality, dict) else {}
    prices = context_pack.get("prices", {}) if isinstance(context_pack, dict) else {}
    bars = prices.get("bars", []) if isinstance(prices, dict) else []
    notes = data_quality.get("notes", []) if isinstance(data_quality, dict) else []
    notes_upper = [str(note).upper() for note in notes]

    reasons: list[str] = []
    age_minutes = _derive_age_minutes(context_pack=context_pack, prices_quality=prices_quality)

    stale = bool(prices_quality.get("stale", False))
    if stale:
        _push_reason(reasons, "STALE_DATA")

    if len(bars) < 10:
        _push_reason(reasons, "INSUFFICIENT_BARS")

    if any("SUSPECT_SERIES" in note for note in notes_upper):
        _push_reason(reasons, "SUSPECT_SERIES")

    if any("INSUFFICIENT_BARS" in note for note in notes_upper):
        _push_reason(reasons, "INSUFFICIENT_BARS")

    if any(("MISSING" in note and ("PRICE" in note or "INDICATOR" in note)) for note in notes_upper):
        _push_reason(reasons, "MISSING_FIELDS")

    if not bars:
        _push_reason(reasons, "MISSING_PRICES")

    show_tool_down = _detect_price_tool_down(
        notes_upper=notes_upper,
        prices_quality=prices_quality,
        prices=prices,
    )
    if show_tool_down:
        _push_reason(reasons, "PRICE_TOOL_DOWN")

    show_degraded = bool(
        stale
        or show_tool_down
        or "INSUFFICIENT_BARS" in reasons
        or "SUSPECT_SERIES" in reasons
        or "MISSING_FIELDS" in reasons
        or "MISSING_PRICES" in reasons
    )

    return {
        "age_minutes": age_minutes,
        "show_degraded": show_degraded,
        "show_tool_down": show_tool_down,
        "reasons": reasons,
    }


def _derive_age_minutes(context_pack: dict, prices_quality: dict) -> int:
    age_raw = prices_quality.get("age_minutes") if isinstance(prices_quality, dict) else None
    age_val = _to_float(age_raw)
    if age_val is None:
        prices_as_of = _to_str(context_pack.get("prices", {}).get("as_of"))
        generated_at = _to_str(context_pack.get("meta", {}).get("generated_at"))
        if prices_as_of and generated_at:
            try:
                age_val = minutes_between(prices_as_of, generated_at)
            except Exception:
                age_val = 0.0
        else:
            age_val = 0.0

    if age_val is None or math.isnan(age_val):
        age_val = 0.0
    return max(0, int(age_val))


def _detect_price_tool_down(notes_upper: list[str], prices_quality: dict, prices: dict) -> bool:
    if isinstance(prices_quality, dict) and prices_quality.get("error"):
        return True
    if isinstance(prices, dict) and prices.get("error"):
        return True

    failure_tokens = [
        "TOOL_DOWN",
        "PRICE_FETCH_FAILED",
        "PRICE_PROVIDER_DOWN",
        "PROVIDER_DOWN",
        "CONTEXT GENERATION UNAVAILABLE",
        "DEGRADED_UI",
    ]
    for note in notes_upper:
        if any(token in note for token in failure_tokens):
            return True
    return False


def _push_reason(reasons: list[str], value: str) -> None:
    if value not in reasons:
        reasons.append(value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
