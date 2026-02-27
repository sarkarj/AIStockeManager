from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceSanityResult:
    display_price: float | None
    source: str  # bars | quote | none
    quality_flags: set[str]
    note: str | None


def reconcile_price_last(
    *,
    ticker: str,
    indicator_price_last: float | None,
    primary_series_close: float | None,
    fallback_series_close: float | None,
) -> PriceSanityResult:
    _ = ticker  # Reserved for future diagnostics; keep deterministic and side-effect free.

    flags: set[str] = set()
    note: str | None = None

    bars_close: float | None = _to_float_or_none(primary_series_close)
    if bars_close is not None:
        source = "bars"
        display_price = bars_close
    else:
        fallback_close = _to_float_or_none(fallback_series_close)
        if fallback_close is not None:
            source = "bars"
            display_price = fallback_close
            flags.add("MISSING_BARS")
            bars_close = fallback_close
        else:
            source = "none"
            display_price = None
            flags.add("MISSING_BARS")

    indicator_close = _to_float_or_none(indicator_price_last)
    if indicator_close is not None and bars_close is not None and bars_close != 0.0:
        mismatch_ratio = abs(indicator_close - bars_close) / abs(bars_close)
        if mismatch_ratio >= 0.20:
            flags.add("PRICE_MISMATCH")
            note = (
                "indicator_price_last "
                f"{indicator_close:.4f} differs from bars_close {bars_close:.4f} "
                f"({mismatch_ratio * 100:.1f}% delta)"
            )

    return PriceSanityResult(
        display_price=display_price,
        source=source,
        quality_flags=flags,
        note=note,
    )


def _to_float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
