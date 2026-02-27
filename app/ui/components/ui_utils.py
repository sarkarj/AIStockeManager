from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def format_money(x: Any) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        value = 0.0
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def format_pct(x: Any) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        value = 0.0
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def action_to_ui_label(action: Any) -> str:
    mapping = {
        "ACCUMULATE": "BUY",
        "WAIT": "HOLD",
        "REDUCE": "SELL",
    }
    return mapping.get(str(action), "HOLD")


def action_to_pill_class(action: Any) -> str:
    mapping = {
        "ACCUMULATE": "pill-buy",
        "WAIT": "pill-hold",
        "REDUCE": "pill-sell",
    }
    return mapping.get(str(action), "pill-hold")


def compute_sparkline_series(bars: Sequence[dict], n: int = 30) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for bar in bars[-n:]:
        if not isinstance(bar, dict):
            continue
        ts = str(bar.get("ts", ""))
        try:
            close = float(bar.get("close", 0.0))
        except (TypeError, ValueError):
            continue
        points.append((ts, close))
    return points


def safe_get(d: Any, path: str | Sequence[str], default: Any = None) -> Any:
    if not isinstance(path, str) and not isinstance(path, Sequence):
        return default
    keys = path.split(".") if isinstance(path, str) else [str(k) for k in path]
    cursor = d
    for key in keys:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            return default
    return cursor
