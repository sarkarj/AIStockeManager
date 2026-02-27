from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.drl.drl_engine import evaluate_drl
from app.core.orchestration.time_utils import now_iso
from app.ui.utils.df_safe import df_for_streamlit

NOW_ISO = "2026-02-11T12:00:00-05:00"


def run_invariants_quickcheck(policy_path: str) -> dict:
    checks: list[dict] = []

    inv1_inputs = {
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
    inv1_result = evaluate_drl(policy_path=policy_path, inputs=inv1_inputs, now_iso=NOW_ISO)
    inv1_ok = inv1_result["action_final"] != "ACCUMULATE"
    checks.append(
        {
            "id": "INV1",
            "passed": inv1_ok,
            "details": "Oversold without strong BULL/BULL exception must not ACCUMULATE.",
        }
    )

    inv2_inputs = {
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
    inv2_result = evaluate_drl(policy_path=policy_path, inputs=inv2_inputs, now_iso=NOW_ISO)
    inv2_ok = inv2_result["action_final"] != "ACCUMULATE"
    checks.append(
        {
            "id": "INV2",
            "passed": inv2_ok,
            "details": "Overbought states must not ACCUMULATE.",
        }
    )

    inv3_inputs = {
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
    inv3_result = evaluate_drl(policy_path=policy_path, inputs=inv3_inputs, now_iso=NOW_ISO)
    inv3_ok = inv3_result["action_final"] == "WAIT" and "TIMEFRAME_CONFLICT" in inv3_result.get("conflicts", [])
    checks.append(
        {
            "id": "INV3",
            "passed": inv3_ok,
            "details": "Timeframe conflict (1D vs 1W) must force WAIT and include TIMEFRAME_CONFLICT.",
        }
    )

    inv4_inputs = {
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
    inv4_result = evaluate_drl(policy_path=policy_path, inputs=inv4_inputs, now_iso=NOW_ISO)
    inv4_ok = float(inv4_result["confidence_cap"]) <= 55.0
    checks.append(
        {
            "id": "INV4",
            "passed": inv4_ok,
            "details": "ATR EXTREME must cap confidence at or below 55.",
        }
    )

    inv5_inputs = {
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
    inv5_result = evaluate_drl(policy_path=policy_path, inputs=inv5_inputs, now_iso=NOW_ISO)
    inv5_ok = (
        inv5_result["action_final"] == "WAIT"
        and "STALE_DATA" in inv5_result.get("conflicts", [])
        and float(inv5_result["confidence_cap"]) <= 55.0
    )
    checks.append(
        {
            "id": "INV5",
            "passed": inv5_ok,
            "details": "Stale inputs must force WAIT, include STALE_DATA, and cap confidence <= 55.",
        }
    )

    overall_ok = all(bool(check["passed"]) for check in checks)
    return {
        "ok": overall_ok,
        "timestamp": now_iso(),
        "checks": checks,
    }


def render_invariants_dashboard(policy_path: str) -> None:
    st.title("Invariants")

    if "invariants_last_result" not in st.session_state:
        st.session_state["invariants_last_result"] = run_invariants_quickcheck(policy_path)

    if st.button("Run Invariants Quickcheck"):
        st.session_state["invariants_last_result"] = run_invariants_quickcheck(policy_path)

    result = st.session_state.get("invariants_last_result", {})
    timestamp = result.get("timestamp", "unknown")

    st.caption(f"Last run: {timestamp}")
    if result.get("ok"):
        st.success("All invariants passed.")
    else:
        st.error("One or more invariants failed.")

    checks = result.get("checks", [])
    if checks:
        rows = [
            {
                "Invariant": check.get("id", ""),
                "Status": "PASS" if check.get("passed") else "FAIL",
                "Details": check.get("details", ""),
            }
            for check in checks
        ]
        st.dataframe(df_for_streamlit(pd.DataFrame(rows)), width="stretch", hide_index=True)
