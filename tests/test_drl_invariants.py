from __future__ import annotations

import pathlib

from app.core.drl.drl_engine import evaluate_drl

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"
NOW_ISO = "2026-02-11T12:00:00-05:00"


def _eval(inputs: dict) -> dict:
    return evaluate_drl(policy_path=str(POLICY_PATH), inputs=inputs, now_iso=NOW_ISO)


def test_inv1_oversold_avoids_accumulate_without_exception() -> None:
    inputs = {
        "ticker": "INV1",
        "as_of": NOW_ISO,
        "price_last": 311.0,
        "ema_50": 318.0,
        "sma_200": 242.0,
        "rsi_14": 28.0,
        "macd": -1.0,
        "macd_signal": 0.0,
        "stoch_k": 10.0,
        "adx_14": 16.0,
        "vroc_14": 5.0,
        "atr_pct": 3.5,
        "supertrend_dir_1D": "BEAR",
        "supertrend_dir_1W": "BEAR",
    }
    result = _eval(inputs)

    assert result["action_final"] != "ACCUMULATE"
    assert result["action_final"] == "WAIT"


def test_inv2_overbought_never_accumulate() -> None:
    inputs = {
        "ticker": "INV2",
        "as_of": NOW_ISO,
        "price_last": 340.0,
        "ema_50": 320.0,
        "sma_200": 250.0,
        "rsi_14": 74.0,
        "macd": 5.0,
        "macd_signal": 3.0,
        "stoch_k": 97.0,
        "adx_14": 34.0,
        "vroc_14": 22.0,
        "atr_pct": 2.0,
        "supertrend_dir_1D": "BULL",
        "supertrend_dir_1W": "BULL",
    }
    result = _eval(inputs)

    assert result["action_final"] != "ACCUMULATE"
    assert result["action_final"] == "WAIT"


def test_inv3_timeframe_conflict_forces_wait_and_conflict_tag() -> None:
    inputs = {
        "ticker": "INV3",
        "as_of": NOW_ISO,
        "price_last": 305.0,
        "ema_50": 318.0,
        "sma_200": 250.0,
        "rsi_14": 40.0,
        "macd": -1.0,
        "macd_signal": 0.0,
        "stoch_k": 35.0,
        "adx_14": 22.0,
        "vroc_14": 10.0,
        "atr_pct": 3.0,
        "supertrend_dir_1D": "BEAR",
        "supertrend_dir_1W": "BULL",
    }
    result = _eval(inputs)

    assert result["regime_1D"] != result["regime_1W"]
    assert result["regime_1W"] != "NEUTRAL"
    assert result["action_final"] == "WAIT"
    assert "TIMEFRAME_CONFLICT" in result["conflicts"]


def test_inv4_atr_extreme_caps_confidence() -> None:
    inputs = {
        "ticker": "INV4",
        "as_of": NOW_ISO,
        "price_last": 200.0,
        "ema_50": 210.0,
        "sma_200": 220.0,
        "rsi_14": 50.0,
        "macd": -0.5,
        "macd_signal": -0.2,
        "stoch_k": 50.0,
        "adx_14": 20.0,
        "vroc_14": 5.0,
        "atr_pct": 6.5,
        "supertrend_dir_1D": "BEAR",
        "supertrend_dir_1W": "BEAR",
    }
    result = _eval(inputs)

    assert float(result["confidence_cap"]) <= 55.0


def test_inv5_stale_forces_wait_and_stale_conflict() -> None:
    inputs = {
        "ticker": "INV5",
        "as_of": "2026-02-10T06:00:00-05:00",
        "price_last": 300.0,
        "ema_50": 295.0,
        "sma_200": 250.0,
        "rsi_14": 55.0,
        "macd": 1.0,
        "macd_signal": 0.8,
        "stoch_k": 50.0,
        "adx_14": 22.0,
        "vroc_14": 10.0,
        "atr_pct": 3.0,
        "supertrend_dir_1D": "BULL",
        "supertrend_dir_1W": "BULL",
    }
    result = _eval(inputs)

    assert result["action_final"] == "WAIT"
    assert "STALE_DATA" in result["conflicts"]
    assert float(result["confidence_cap"]) <= 55.0
