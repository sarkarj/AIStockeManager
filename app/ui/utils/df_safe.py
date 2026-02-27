from __future__ import annotations

import pandas as pd


def df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make df Arrow-safe for Streamlit by preventing mixed/object type conversion failures.
    Deterministic: preserves values as display strings in problematic columns only.
    """
    out = df.copy()

    # 1) Normalize columns with dtype "object" to safe strings (most common failure source)
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(lambda x: "" if x is None else str(x))

    # 2) Convert sets/dicts/lists (often appear in quality_flags) to stable strings
    def _stringify(x):
        if isinstance(x, (set, dict, list, tuple)):
            return str(x)
        return x

    # Apply stringify per column (avoid applymap)
    for col in out.columns:
        out[col] = out[col].map(_stringify)

    # 3) Final pass: force any remaining object columns to str
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)

    return out
