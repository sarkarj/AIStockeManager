from __future__ import annotations

import numpy as np
import pandas as pd


def compute_required_metrics_from_prices(prices: dict) -> dict:
    bars = prices.get("bars", [])
    if not isinstance(bars, list) or not bars:
        raise ValueError("prices.bars must be a non-empty list")

    frame = pd.DataFrame(bars)
    required_cols = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing bar fields: {missing}")

    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    frame = frame.sort_values("ts").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].ffill().bfill()
    frame["volume"] = frame["volume"].fillna(0.0)

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    ema_50 = close.ewm(span=50, adjust=False).mean()
    sma_200 = close.rolling(window=200, min_periods=1).mean()

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    rsi = rsi.fillna(50.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    stoch_den = (high_14 - low_14).replace(0.0, np.nan)
    stoch_k = ((close - low_14) / stoch_den) * 100.0
    stoch_k = stoch_k.clip(0.0, 100.0).fillna(50.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up_move = high.diff().fillna(0.0)
    down_move = (-low.diff()).fillna(0.0)
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index)

    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    adx_14 = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0.0)

    vol_shift = volume.shift(14)
    vroc_14 = ((volume - vol_shift) / vol_shift.replace(0.0, np.nan)) * 100.0
    vroc_14 = vroc_14.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    atr_pct = (atr / close.replace(0.0, np.nan)) * 100.0
    atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metrics = {
        "price_last": _last_float(close, default=0.0),
        "ema_50": _last_float(ema_50, default=_last_float(close, default=0.0)),
        "sma_200": _last_float(sma_200, default=_last_float(close, default=0.0)),
        "rsi_14": _last_float(rsi, default=50.0),
        "macd": _last_float(macd, default=0.0),
        "macd_signal": _last_float(macd_signal, default=0.0),
        "stoch_k": _last_float(stoch_k, default=50.0),
        "adx_14": _last_float(adx_14, default=0.0),
        "vroc_14": _last_float(vroc_14, default=0.0),
        "atr_pct": _last_float(atr_pct, default=0.0),
    }
    return metrics


def _last_float(series: pd.Series, default: float) -> float:
    if series.empty:
        return float(default)
    value = series.iloc[-1]
    if pd.isna(value):
        return float(default)
    return float(value)
