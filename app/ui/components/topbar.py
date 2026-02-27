from __future__ import annotations

import re

import streamlit as st

from app.core.marketdata.query_graph import MarketQueryService


_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")


def render_topbar() -> None:
    left, right = st.columns([2.7, 2.2], vertical_alignment="center")

    with left:
        spy = _proxy_summary("SPY")
        qqq = _proxy_summary("QQQ")
        vix = _proxy_summary("^VIX")
        st.markdown(
            f"<div class='tiny muted'>SPY: {spy} | QQQ: {qqq} | VIX: {vix}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.session_state.setdefault("manual_refresh_scope", "visible")
        search_col, open_col, scope_col, refresh_col = st.columns([3.6, 1.1, 1.8, 1.5], vertical_alignment="center")
        with search_col:
            st.text_input(
                "Search or add ticker…",
                key="topbar_ticker_input",
                placeholder="Search or add ticker…",
                label_visibility="collapsed",
                on_change=_set_pending_select,
            )
        with open_col:
            if st.button("Open", key="topbar_open_button", width="stretch"):
                _set_pending_select()
        with scope_col:
            st.selectbox(
                "Refresh scope",
                options=["visible", "brain", "horizon", "all"],
                format_func=_refresh_scope_label,
                key="manual_refresh_scope",
                label_visibility="collapsed",
            )
        with refresh_col:
            if st.button("Refresh Data", key="topbar_refresh_button", width="stretch"):
                st.session_state["manual_refresh_requested"] = True


def _set_pending_select() -> None:
    ticker = _normalize_ticker(st.session_state.get("topbar_ticker_input", ""))
    if not ticker:
        return
    st.session_state["pending_select_ticker"] = ticker
    st.session_state["topbar_input_error"] = ""


def _normalize_ticker(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        st.session_state["topbar_input_error"] = "Enter a ticker symbol."
        return None
    if not _TICKER_PATTERN.fullmatch(text):
        st.session_state["topbar_input_error"] = "Use only A-Z, 0-9, '.' or '-'."
        return None
    return text


def _proxy_summary(ticker: str) -> str:
    query = st.session_state.get("_market_query")
    if isinstance(query, MarketQueryService):
        try:
            quote = query.quote_snapshot_dict(ticker)
            latest = float(quote.get("latest_price")) if quote.get("latest_price") is not None else None
            prev_close = float(quote.get("prev_close_price")) if quote.get("prev_close_price") is not None else None
            if latest is None or prev_close is None or prev_close == 0:
                return "—"
            delta_pct = ((latest - prev_close) / prev_close) * 100.0
            sign = "+" if delta_pct > 0 else ""
            return f"{latest:.2f} ({sign}{delta_pct:.2f}%)"
        except Exception:
            return "—"

    loader = st.session_state.get("_context_loader")
    try:
        pack = loader(ticker=ticker, generate_hub_card=False, interval="1h", lookback_days=7)
        bars = pack.get("prices", {}).get("bars", [])
        if len(bars) < 2:
            return "—"
        last_close = float(bars[-1].get("close", 0.0))
        prev_close = float(bars[-2].get("close", last_close))
        if prev_close == 0:
            return "—"
        delta_pct = ((last_close - prev_close) / prev_close) * 100.0
        sign = "+" if delta_pct > 0 else ""
        return f"{last_close:.2f} ({sign}{delta_pct:.2f}%)"
    except Exception:
        return "—"


def _refresh_scope_label(value: str) -> str:
    labels = {
        "visible": "Visible",
        "brain": "Brain Only",
        "horizon": "Horizon Only",
        "all": "Queue All",
    }
    return labels.get(str(value), str(value))
