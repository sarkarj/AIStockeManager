from __future__ import annotations

import math
from typing import Optional, Union

Number = Union[int, float]


def format_money(value: Optional[Number], *, currency: str = "USD", decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    symbol = "$" if currency == "USD" else ""
    return f"{symbol}{v:,.{decimals}f}"


def format_int(value: Optional[Number]) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{int(round(v)):,}"
