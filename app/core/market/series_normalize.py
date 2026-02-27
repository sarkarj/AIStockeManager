from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_bars_for_chart(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(bars, list) or not bars:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(bars)
    required = {"ts", "open", "high", "low", "close"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    else:
        frame["volume"] = pd.NA

    frame = frame.dropna(subset=["ts", "open", "high", "low", "close"])
    if frame.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame = frame.sort_values("ts")
    frame = frame.drop_duplicates(subset=["ts"], keep="last")
    frame = frame.reset_index(drop=True)

    if not frame["ts"].is_monotonic_increasing:
        raise ValueError("Non-monotonic timestamps after normalization.")

    return frame[["ts", "open", "high", "low", "close", "volume"]]
