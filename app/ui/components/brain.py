from __future__ import annotations

import html
import json
import re
import time
import warnings
from datetime import timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app.core.context_pack.why_cache import (
    WhyArtifact,
    build_why_signature,
    hydrate_context_pack_with_why,
    load_latest_why_artifact,
    load_why_artifact,
    save_why_artifact,
)
from app.core.context_pack.hub_generator import (
    generate_hub_for_context_pack,
    is_llm_configured as hub_is_llm_configured,
    resolve_bedrock_config,
)
from app.core.marketdata.chart_fetcher import ChartFetcher, range_mapping
from app.core.marketdata.prewarm import cache_hygiene_snapshot, enqueue_prewarm_request, load_prewarm_status, prewarm_queue_depth
from app.core.marketdata.query_graph import MarketQueryService
from app.core.query.contracts import LongQueryResult, ShortQueryResult, run_long_query, run_short_query
from app.core.market.series_normalize import normalize_bars_for_chart
from app.core.orchestration.time_utils import now_iso, parse_iso
from app.ui.components.brain_market_card import render_brain_market_card
from app.ui.components.invariants_dashboard import run_invariants_quickcheck
from app.ui.components.replay_panel import render_replay_tools
from app.ui.components.trust_badges import compute_brain_trust_state, render_badges
from app.ui.components.ui_utils import format_money, safe_get
from app.ui.utils.df_safe import df_for_streamlit
from app.ui.utils.exports import rows_to_csv_bytes, to_json_bytes
from app.ui.viewmodels.brain_market_vm import BrainMarketVM, build_brain_market_vm
from app.ui.viewmodels.brain_vm import build_brain_view_model

RANGE_ORDER = ["1D", "1W", "1M", "3M", "YTD", "1Y", "Advanced"]
NON_ADVANCED_RANGES = ["1D", "1W", "1M", "3M", "YTD", "1Y"]

RANGE_VIEW_META: dict[str, dict[str, Any]] = {
    "1D": {"label": "1D (RTH+Extended)", "min_points": 6, "max_points": 128},
    "1W": {"label": "1W", "min_points": 10, "max_points": 256},
    "1M": {"label": "1M", "min_points": 10, "max_points": 512},
    "3M": {"label": "3M", "min_points": 10, "max_points": 512},
    "YTD": {"label": "YTD", "min_points": 10, "max_points": 512},
    "1Y": {"label": "1Y", "min_points": 10, "max_points": 512},
}


def brain_sections() -> list[str]:
    return ["Header", "Charts", "The Why", "View Evidence", "Diagnostics"]


def get_range_contract(range_key: str, advanced_source: str | None = None) -> dict[str, Any]:
    key = range_key if range_key in NON_ADVANCED_RANGES else "3M"
    if range_key == "Advanced":
        key = advanced_source if advanced_source in NON_ADVANCED_RANGES else "3M"
    mapping = range_mapping(key)
    base = RANGE_VIEW_META.get(key, RANGE_VIEW_META["3M"])
    return {
        "range_key": range_key,
        "source_range": key,
        "period": mapping["period"],
        "interval": mapping["interval"],
        "prepost": bool(mapping["prepost"]),
        "expected_tz": "America/New_York",
        "label": "Advanced" if range_key == "Advanced" else str(base["label"]),
        "min_points": int(base["min_points"]),
        "max_points": int(base["max_points"]),
    }


def render_brain(
    selected_ticker: str | None,
    context_loader,
    policy_path: str,
    market_query: MarketQueryService | None = None,
) -> None:
    st.markdown("<div class='card'><div class='section-title'>The Brain</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='pulse-block-gap'></div>", unsafe_allow_html=True)
    if not selected_ticker:
        st.info("Select a ticker from The Pulse or The Horizon to open Brain view.")
        return

    query = market_query or MarketQueryService(cache_dir=".cache/charts", context_loader=context_loader)
    generate_hub = _is_llm_configured()
    symbol = str(selected_ticker or "").strip().upper()
    hydration = _brain_hydration_entry(ticker=symbol)

    if hydration.get("short_result") is None:
        short_started = time.perf_counter()
        try:
            hydration["short_result"] = run_short_query(ticker=symbol, market_query=query)
            hydration["stage1_ms"] = round((time.perf_counter() - short_started) * 1000.0, 2)
            hydration["phase"] = "stage1_ready"
            hydration["error"] = ""
            _set_brain_hydration_entry(ticker=symbol, entry=hydration)
        except Exception:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.warning("Ticker details are temporarily unavailable. Showing degraded view.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    short_result = hydration.get("short_result")
    if not isinstance(short_result, ShortQueryResult):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.warning("Ticker details are temporarily unavailable. Showing degraded view.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if hydration.get("phase") == "stage1_ready":
        _render_stage1_brain(symbol=symbol, short_result=short_result)
        hydration["phase"] = "stage2_fetch"
        _set_brain_hydration_entry(ticker=symbol, entry=hydration)
        st.rerun()
        return

    if hydration.get("phase") == "stage2_fetch":
        stage2_started = time.perf_counter()
        try:
            hydration["long_result"] = run_long_query(
                ticker=symbol,
                range_key="1D",
                include_why=False,
                market_query=query,
            )
            hydration["stage2_ms"] = round((time.perf_counter() - stage2_started) * 1000.0, 2)
            hydration["phase"] = "ready"
            hydration["error"] = ""
            _set_brain_hydration_entry(ticker=symbol, entry=hydration)
            st.rerun()
            return
        except Exception as exc:
            hydration["phase"] = "degraded"
            hydration["error"] = str(exc)[:160]
            _set_brain_hydration_entry(ticker=symbol, entry=hydration)

    long_result = hydration.get("long_result")
    if not isinstance(long_result, LongQueryResult):
        _render_stage1_brain(symbol=symbol, short_result=short_result, degraded_reason=str(hydration.get("error", "") or ""))
        return

    context_pack = long_result.context_pack
    quote = long_result.quote if isinstance(long_result.quote, dict) else {}
    if not isinstance(quote, dict):
        quote = {}

    why_signature = str(long_result.why_signature or "").strip().lower()
    why_cache_state = "disabled"
    why_sync_status = "not_attempted"
    why_sync_error = ""
    why_sync_elapsed_ms = 0.0
    why_llm_usage: dict[str, Any] = {}
    if generate_hub and why_signature:
        artifact = load_why_artifact(signature=why_signature, ticker=str(selected_ticker))
        if artifact is not None:
            context_pack = hydrate_context_pack_with_why(context_pack=context_pack, artifact=artifact)
            why_cache_state = "cache_hit"
        else:
            sync_result = _attempt_sync_why_generation(
                ticker=str(selected_ticker),
                why_signature=why_signature,
                range_key="1D",
                context_pack=context_pack,
                quote=quote,
                timeout_seconds=8.0,
            )
            why_sync_status = str(sync_result.get("status", "not_attempted"))
            why_sync_error = str(sync_result.get("error", "") or "")
            why_sync_elapsed_ms = float(sync_result.get("elapsed_ms", 0.0) or 0.0)
            why_llm_usage = (
                dict(sync_result.get("llm_usage", {}))
                if isinstance(sync_result.get("llm_usage"), dict)
                else {}
            )
            sync_artifact = sync_result.get("artifact")
            if isinstance(sync_artifact, WhyArtifact):
                context_pack = hydrate_context_pack_with_why(context_pack=context_pack, artifact=sync_artifact)
                why_cache_state = "live_sync"
            else:
                latest_artifact = load_latest_why_artifact(ticker=str(selected_ticker), max_age_minutes=360)
                if latest_artifact is not None:
                    context_pack = hydrate_context_pack_with_why(context_pack=context_pack, artifact=latest_artifact)
                    why_cache_state = "cache_fallback"
            queued = _enqueue_why_refresh(
                ticker=str(selected_ticker),
                why_signature=why_signature,
                range_key="1D",
            )
            if why_cache_state == "cache_fallback":
                why_cache_state = "cache_fallback_queued" if queued else "cache_fallback"
            elif why_cache_state == "live_sync":
                why_cache_state = "live_sync_queued" if queued else "live_sync"
            else:
                why_cache_state = "queued" if queued else "pending"
                reason = "Why narrative refreshing in background."
                if why_sync_status in {"timeout", "error"} and why_sync_error:
                    reason = f"{reason} sync attempt failed: {why_sync_error}"
                _set_hub_refresh_reason(context_pack=context_pack, reason=reason)
    elif not generate_hub:
        why_cache_state = "llm_not_configured"
        why_sync_status = "llm_not_configured"
        _set_hub_refresh_reason(context_pack=context_pack, reason="LLM not configured")
    _attach_why_meta(
        context_pack=context_pack,
        why_signature=why_signature or None,
        why_cache_state=why_cache_state,
        why_sync_status=why_sync_status,
        why_sync_error=why_sync_error,
        why_sync_elapsed_ms=why_sync_elapsed_ms,
        why_llm_usage=why_llm_usage,
    )

    chart_fetcher = _get_chart_fetcher()
    primary_series = _get_price_series_for_range(
        ticker=selected_ticker,
        range_key="1D",
        fetcher=chart_fetcher,
        market_query=query,
    )
    if str(quote.get("source", "none")).lower() in {"", "none"}:
        fallback_quote = safe_get(context_pack, "meta.latest_quote", {})
        if isinstance(fallback_quote, dict):
            quote = dict(fallback_quote)

    primary_close = _series_last_close(primary_series)
    vm = build_brain_view_model(
        context_pack,
        quote=quote,
        primary_series_close=primary_close,
        series_for_selected_range=primary_series,
    )
    price_sanity_flags = [str(x) for x in safe_get(vm, "price_sanity.quality_flags", []) or []]
    if price_sanity_flags:
        primary_flags = [str(x) for x in primary_series.get("flags", [])]
        primary_series["flags"] = _unique(primary_flags + price_sanity_flags)

    action_raw = str(vm.get("drl_action_raw", "WAIT"))
    action_ui = str(vm.get("ui_action_label", "HOLD"))
    confidence = float(vm.get("confidence_cap", 0.0))
    pill_class = str(vm.get("ui_action_pill_class", "pill-hold"))
    chart_error = str(safe_get(primary_series, "diagnostics.error", "") or "")
    quote_source = str(safe_get(vm, "quote.source", "none")).lower()
    market_up = not _is_provider_failure(chart_error) and (quote_source != "none" or primary_series.get("source") != "none")
    trust_state = compute_brain_trust_state(
        context_pack=context_pack,
        chart_series=primary_series,
        market_data_provider_up=market_up,
        quote=safe_get(vm, "quote", {}) or {},
        price_sanity_flags=safe_get(vm, "price_sanity.quality_flags", []) or [],
    )

    _render_header_card(
        selected_ticker,
        context_pack,
        vm,
        action_raw,
        action_ui,
        confidence,
        pill_class,
        trust_state,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    why_block = _build_market_why_block(context_pack=context_pack, vm=vm, primary_series=primary_series)
    series_by_range = _render_market_tabs(
        selected_ticker,
        context_pack,
        trust_state=trust_state,
        why_block=why_block,
        chart_fetcher=chart_fetcher,
        market_query=query,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _render_evidence(context_pack=context_pack, series_by_range=series_by_range, vm=vm)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _render_diagnostics(
        selected_ticker=selected_ticker,
        context_pack=context_pack,
        policy_path=policy_path,
        series_by_range=series_by_range,
        vm=vm,
        hydration_diag={
            "phase": str(hydration.get("phase", "ready")),
            "stage1_ms": float(hydration.get("stage1_ms", 0.0) or 0.0),
            "stage2_ms": float(hydration.get("stage2_ms", 0.0) or 0.0),
        },
    )


def _brain_hydration_entry(*, ticker: str) -> dict[str, Any]:
    store = st.session_state.setdefault("brain_hydration", {})
    if not isinstance(store, dict):
        store = {}
        st.session_state["brain_hydration"] = store
    symbol = str(ticker or "").strip().upper()
    refresh_token = _brain_refresh_token(symbol)
    entry = store.get(symbol)
    if isinstance(entry, dict) and int(entry.get("refresh_token", -1)) == int(refresh_token):
        return entry
    new_entry = {
        "refresh_token": int(refresh_token),
        "phase": "stage1_pending",
        "short_result": None,
        "long_result": None,
        "stage1_ms": 0.0,
        "stage2_ms": 0.0,
        "error": "",
    }
    store[symbol] = new_entry
    return new_entry


def _set_brain_hydration_entry(*, ticker: str, entry: dict[str, Any]) -> None:
    store = st.session_state.setdefault("brain_hydration", {})
    if not isinstance(store, dict):
        store = {}
        st.session_state["brain_hydration"] = store
    symbol = str(ticker or "").strip().upper()
    store[symbol] = dict(entry or {})


def _brain_refresh_token(ticker: str) -> int:
    tokens = st.session_state.get("context_refresh_tokens", {})
    if not isinstance(tokens, dict):
        return 0
    raw = tokens.get(str(ticker or "").strip().upper(), 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _render_stage1_brain(*, symbol: str, short_result: ShortQueryResult, degraded_reason: str = "") -> None:
    context_pack = short_result.context_pack if isinstance(short_result.context_pack, dict) else {}
    quote = short_result.quote if isinstance(short_result.quote, dict) else {}
    series = _chart_series_to_ui_series(series_obj=short_result.series_1d, range_key="1D")
    primary_close = _series_last_close(series)
    vm = build_brain_view_model(
        context_pack,
        quote=quote,
        primary_series_close=primary_close,
        series_for_selected_range=series,
    )
    action_raw = str(vm.get("drl_action_raw", "WAIT"))
    action_ui = str(vm.get("ui_action_label", "HOLD"))
    confidence = float(vm.get("confidence_cap", 0.0))
    pill_class = str(vm.get("ui_action_pill_class", "pill-hold"))
    quote_source = str(safe_get(vm, "quote.source", "none")).lower()
    market_up = quote_source != "none" or int(series.get("point_count", 0)) > 0
    trust_state = compute_brain_trust_state(
        context_pack=context_pack,
        chart_series=series,
        market_data_provider_up=market_up,
        quote=safe_get(vm, "quote", {}) or {},
        price_sanity_flags=safe_get(vm, "price_sanity.quality_flags", []) or [],
    )
    _render_header_card(
        symbol,
        context_pack,
        vm,
        action_raw,
        action_ui,
        confidence,
        pill_class,
        trust_state,
    )
    if degraded_reason:
        st.warning(f"Hydration degraded: {degraded_reason}")
    else:
        st.caption("Hydrating chart ranges, Why narrative, and diagnostics...")


def _chart_series_to_ui_series(*, series_obj: Any, range_key: str) -> dict[str, Any]:
    contract = get_range_contract(range_key=range_key)
    if not hasattr(series_obj, "bars"):
        return {
            "range_key": range_key,
            "bars": [],
            "as_of": now_iso(),
            "source": "none",
            "point_count": 0,
            "min_points": int(contract.get("min_points", 0)),
            "flags": ["MISSING"],
            "stale": False,
            "synthetic": False,
            "contract": contract,
            "diagnostics": {"error": None, "source": "none"},
        }
    bars = []
    for bar in list(getattr(series_obj, "bars", []) or []):
        bars.append(
            {
                "ts": bar.ts.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": None if bar.volume is None else float(bar.volume),
            }
        )
    series = build_chart_series_from_bars(
        range_key=range_key,
        bars=bars,
        as_of=getattr(series_obj, "as_of", None).isoformat() if getattr(series_obj, "as_of", None) else now_iso(),
        source=str(getattr(series_obj, "source", "none")),
        now_iso_value=now_iso(),
        min_points=int(contract.get("min_points", 0)),
        max_points=int(contract.get("max_points", 128)),
    )
    base_flags = list(series.get("flags", []))
    quality_flags = [str(x) for x in (getattr(series_obj, "quality_flags", set()) or set())]
    series["flags"] = _unique(base_flags + quality_flags)
    series["contract"] = contract
    series["diagnostics"] = {
        "error": getattr(series_obj, "error", None),
        "source": str(getattr(series_obj, "source", "none")),
        "cache_hit": bool(getattr(series_obj, "cache_hit", False)),
        "cache_age_minutes": getattr(series_obj, "cache_age_minutes", None),
        "fetch_ms": 0.0,
    }
    return series


def _render_header_card(
    ticker: str,
    context_pack: dict,
    vm: dict[str, Any],
    action_raw: str,
    action_ui: str,
    confidence: float,
    pill_class: str,
    trust_state: dict,
) -> None:
    last_price = vm.get("last_price")
    price_source = safe_get(vm, "quote.source", "none") or safe_get(vm, "price_sanity.source", "none")
    quote_close = safe_get(vm, "quote.close_price", None)
    quote_close_ts = safe_get(vm, "quote.close_ts", None)
    quote_close_ts_local = safe_get(vm, "quote.close_ts_local", None)
    quote_after = safe_get(vm, "quote.after_hours_price", None)
    quote_after_ts = safe_get(vm, "quote.after_hours_ts", None)
    quote_after_ts_local = safe_get(vm, "quote.after_hours_ts_local", None)
    show_extended = bool(safe_get(vm, "quote.show_extended_session", False))
    extended_label = str(safe_get(vm, "quote.extended_label", "") or "").strip() or "After-hours"
    extended_icon = "🌙" if extended_label == "After-hours" else "🌅"
    data_as_of_local, _ = _format_as_of_local_utc(safe_get(context_pack, "indicators.as_of", None))
    close_local = str(quote_close_ts_local or "").strip() or _format_as_of_local_utc(quote_close_ts)[0]
    after_local = str(quote_after_ts_local or "").strip() or _format_as_of_local_utc(quote_after_ts)[0]
    freshness = safe_get(context_pack, "meta.data_quality.prices", {}) or {}
    age_minutes = freshness.get("age_minutes")
    age_text = f"{max(0.0, float(age_minutes)):.0f}m" if isinstance(age_minutes, (int, float)) else "unknown"
    price_text = "--" if not isinstance(last_price, (int, float)) else format_money(last_price)
    close_text = format_money(float(quote_close)) if isinstance(quote_close, (int, float)) else "n/a"
    after_text = format_money(float(quote_after)) if isinstance(quote_after, (int, float)) else "n/a"
    today_abs = safe_get(vm, "quote.today_change_abs", None)
    today_pct = safe_get(vm, "quote.today_change_pct", None)
    ah_abs = safe_get(vm, "quote.after_hours_change_abs", None)
    ah_pct = safe_get(vm, "quote.after_hours_change_pct", None)
    today_line = _format_change_line(today_abs, today_pct, label="Today")
    after_change_line = _format_change_line(ah_abs, ah_pct, label=extended_label)

    with st.container(border=True):
        left, right = st.columns([2.2, 1.4], vertical_alignment="center")
        with left:
            st.markdown(
                (
                    "<div class='row'>"
                    f"<div class='section-title'>{html.escape(str(ticker))}</div>"
                    f"<div class='brain-header-price'>{html.escape(str(price_text))}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                (
                    "<div class='tiny muted'>"
                    f"Close: {html.escape(close_text)}"
                    f"{f' · {html.escape(close_local)}' if close_local != 'unknown' else ''}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if show_extended and isinstance(quote_after, (int, float)):
                st.markdown(
                    (
                        "<div class='tiny muted'>"
                        f"{extended_icon} {html.escape(extended_label)}: {html.escape(after_text)}"
                        f"{f' · {html.escape(after_local)}' if after_local != 'unknown' else ''}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            elif show_extended:
                st.markdown(
                    f"<div class='tiny muted'>{extended_icon} {html.escape(extended_label)}: —</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div class='tiny muted'>{html.escape(today_line)}</div>",
                unsafe_allow_html=True,
            )
            if show_extended:
                st.markdown(
                    f"<div class='tiny muted'>{html.escape(after_change_line)}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div class='tiny muted'>Data as_of: {data_as_of_local} · age {age_text} · source {price_source}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:8px 0 2px 0;'>", unsafe_allow_html=True)
        with right:
            st.markdown(
                f"<div class='row-right'><span class='pill {pill_class}'>{action_ui}</span></div>",
                unsafe_allow_html=True,
            )
            confidence_tooltip = (
                "Confidence&#10;"
                "• Confidence summarizes how reliable the DRL action is for the current data + timeframe.&#10;"
                "• It can be capped by policy gates (e.g., oversold-bounce risk, timeframe conflict, stale/missing data).&#10;"
                "• Computed from deterministic indicator signals (EMA/SMA, RSI, MACD, Stoch, ADX, VROC, ATR%) + DRL gates/conflicts; it is NOT a prediction of returns.&#10;"
                "Higher = fewer conflicts + cleaner signals; lower = more caps/risks."
            )
            conf_text = f"Confidence {confidence:.0f}/100"
            if confidence < 100:
                conf_text = f"{conf_text} (capped)"
            st.markdown(
                (
                    "<div class='tiny muted'>"
                    f"{html.escape(conf_text)} "
                    f"<span title='{confidence_tooltip}' style='cursor:help;'>ⓘ</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='tiny muted'>DRL: {action_raw} <span class='muted'>(deterministic)</span></div>",
                unsafe_allow_html=True,
            )
            render_badges(trust_state)
            st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:8px 0 2px 0;'>", unsafe_allow_html=True)


def _render_market_tabs(
    ticker: str,
    context_pack: dict,
    trust_state: dict[str, bool],
    why_block: dict[str, Any],
    chart_fetcher: ChartFetcher | None = None,
    market_query: MarketQueryService | None = None,
) -> dict[str, dict[str, Any]]:
    series_by_range: dict[str, dict[str, Any]] = {}
    fetcher = chart_fetcher or _get_chart_fetcher()
    series_resolver = (
        (lambda rk: market_query.chart_series(ticker=ticker, range_key=rk))
        if isinstance(market_query, MarketQueryService)
        else None
    )
    selector_key = f"brain_active_range_{ticker}"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = "1D"
    active_range = st.radio(
        "Range",
        RANGE_ORDER,
        key=selector_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    range_key = str(active_range)
    advanced_source = None
    if range_key == "Advanced":
        advanced_source = st.selectbox(
            "Advanced source range",
            NON_ADVANCED_RANGES,
            index=NON_ADVANCED_RANGES.index("3M"),
            key=f"brain_advanced_source_{ticker}",
        )

    started = time.perf_counter()
    market_vm = build_brain_market_vm(
        context_pack=context_pack,
        selected_range_key=range_key,
        advanced_source_range=advanced_source,
        chart_fetcher=fetcher,
        series_resolver=series_resolver,
        badge_state=trust_state,
        why_block=why_block,
    )
    fetch_ms = round((time.perf_counter() - started) * 1000.0, 2)
    render_brain_market_card(vm=market_vm, range_key=range_key)
    series_by_range[range_key] = _market_vm_series_to_dict(
        vm=market_vm,
        range_key=range_key,
        fetch_ms=fetch_ms,
    )

    return series_by_range


def _market_vm_series_to_dict(vm: BrainMarketVM, range_key: str, fetch_ms: float) -> dict[str, Any]:
    bars = [
        {
            "ts": bar.ts.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": None if bar.volume is None else float(bar.volume),
        }
        for bar in vm.chart_series.bars
    ]
    contract = get_range_contract(range_key=range_key, advanced_source=vm.chart_source_range)
    flags = list(sorted(vm.chart_series.quality_flags))
    min_points = int(contract.get("min_points", 0))
    if len(bars) == 0 and "MISSING" not in flags:
        flags.append("MISSING")
    if 0 < len(bars) < min_points and "INSUFFICIENT" not in flags:
        flags.append("INSUFFICIENT")
    measured_fetch_ms = 0.0 if vm.chart_series.cache_hit else float(fetch_ms)
    return {
        "range_key": range_key,
        "bars": bars,
        "as_of": vm.chart_series.as_of.isoformat(),
        "source": vm.chart_series.source,
        "point_count": len(bars),
        "min_points": min_points,
        "flags": _unique(flags),
        "stale": any(flag in {"STALE_CACHE", "STALE_MARKET_TS"} for flag in flags),
        "synthetic": False,
        "contract": contract,
        "diagnostics": {
            "cache_key": Path(str(vm.chart_series.cache_path)).name if vm.chart_series.cache_path else "",
            "cache_path": str(vm.chart_series.cache_path),
            "cache_hit": bool(vm.chart_series.cache_hit),
            "cache_age_minutes": (
                None if vm.chart_series.cache_age_minutes is None else max(0.0, float(vm.chart_series.cache_age_minutes))
            ),
            "stale_cache_used": bool(vm.chart_series.stale_cache),
            "attempts": int(vm.chart_series.attempts),
            "fetch_ms": measured_fetch_ms,
            "error": str(vm.chart_series.error) if vm.chart_series.error else None,
            "source": str(vm.chart_series.source),
            "period": contract.get("period"),
            "interval": contract.get("interval"),
            "prepost": bool(contract.get("prepost", False)),
            "expected_tz": contract.get("expected_tz", "America/New_York"),
            "source_range": vm.chart_source_range,
        },
    }


def _render_timeframe_tabs(
    ticker: str,
    context_pack: dict,
    preloaded: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    series_by_range: dict[str, dict[str, Any]] = {}
    tabs = st.tabs(RANGE_ORDER)
    preloaded_series = preloaded or {}

    price_hint = _last_close(safe_get(context_pack, "prices.bars", []) or [])

    for range_key, tab in zip(RANGE_ORDER, tabs):
        with tab:
            if range_key in preloaded_series:
                series = preloaded_series[range_key]
                series_by_range[range_key] = series
                _render_chart_card(ticker=ticker, series=series)
                continue
            if range_key == "Advanced":
                source_range = st.selectbox(
                    "Advanced source range",
                    NON_ADVANCED_RANGES,
                    index=NON_ADVANCED_RANGES.index("3M"),
                    key=f"brain_advanced_source_{ticker}",
                )
                series = _get_price_series_for_range(
                    ticker=ticker,
                    range_key=range_key,
                    advanced_source_range=source_range,
                    price_hint=price_hint,
                )
            else:
                series = _get_price_series_for_range(ticker=ticker, range_key=range_key, price_hint=price_hint)

            series_by_range[range_key] = series
            _render_chart_card(ticker=ticker, series=series)

    return series_by_range


def _get_chart_fetcher() -> ChartFetcher:
    return ChartFetcher(cache_dir=".cache/charts", stale_first=True)


def _get_price_series_for_range(
    ticker: str,
    range_key: str,
    advanced_source_range: str | None = None,
    price_hint: float = 0.0,
    fetcher: ChartFetcher | None = None,
    market_query: MarketQueryService | None = None,
) -> dict[str, Any]:
    contract = get_range_contract(range_key=range_key, advanced_source=advanced_source_range)
    fetcher_obj = fetcher or _get_chart_fetcher()

    started = time.perf_counter()
    if isinstance(market_query, MarketQueryService):
        fetched = market_query.chart_series(ticker=ticker, range_key=str(contract["source_range"]))
    else:
        fetched = fetcher_obj.fetch_chart_series(ticker=ticker, range_key=str(contract["source_range"]))
    fetch_ms = (time.perf_counter() - started) * 1000.0

    bars = [
        {
            "ts": bar.ts.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": None if bar.volume is None else float(bar.volume),
        }
        for bar in fetched.bars
    ]

    series = build_chart_series_from_bars(
        range_key=range_key,
        bars=bars,
        as_of=fetched.as_of.isoformat(),
        source=str(fetched.source),
        now_iso_value=now_iso(),
        min_points=int(contract["min_points"]),
        max_points=int(contract["max_points"]),
    )

    flags = list(series.get("flags", []))
    for flag in sorted(fetched.quality_flags):
        if flag not in flags:
            flags.append(flag)
    series["flags"] = flags
    series["stale"] = any(flag in {"STALE_CACHE", "STALE_MARKET_TS"} for flag in flags)
    series["contract"] = contract
    series["diagnostics"] = {
        "cache_key": Path(str(fetched.cache_path)).name,
        "cache_path": str(fetched.cache_path),
        "cache_hit": bool(fetched.cache_hit),
        "cache_age_minutes": (
            None if fetched.cache_age_minutes is None else max(0.0, float(fetched.cache_age_minutes))
        ),
        "stale_cache_used": bool(fetched.stale_cache),
        "attempts": int(fetched.attempts),
        "fetch_ms": round(fetch_ms, 2),
        "error": str(fetched.error) if fetched.error else None,
        "source": str(fetched.source),
    }
    return series


def build_chart_series_from_bars(
    range_key: str,
    bars: list[dict],
    as_of: str | None,
    source: str,
    now_iso_value: str,
    min_points: int,
    max_points: int,
) -> dict[str, Any]:
    flags: list[str] = []

    frame = _normalize_bars_to_frame(bars=bars, flags=flags)
    frame = _apply_range_filter(frame=frame, range_key=range_key, now_iso_value=now_iso_value)

    if max_points > 0 and len(frame) > max_points:
        frame = frame.tail(max_points)

    bars_out = _frame_to_bars(frame)
    point_count = len(bars_out)

    if point_count == 0 and "MISSING" not in flags:
        flags.append("MISSING")
    if 0 < point_count < int(min_points):
        flags.append("INSUFFICIENT")

    series_as_of = str(as_of) if as_of else (bars_out[-1]["ts"] if bars_out else now_iso_value)
    return {
        "range_key": range_key,
        "bars": bars_out,
        "as_of": series_as_of,
        "source": source,
        "point_count": point_count,
        "min_points": int(min_points),
        "flags": _unique(flags),
        "stale": False,
        "synthetic": False,
    }


def _normalize_bars_to_frame(bars: list[dict], flags: list[str]) -> pd.DataFrame:
    try:
        frame = normalize_bars_for_chart(bars)
    except Exception:
        flags.append("INVALID_TIMESTAMPS")
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    if frame.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return frame


def _apply_range_filter(frame: pd.DataFrame, range_key: str, now_iso_value: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    now_et = parse_iso(now_iso_value).astimezone(ZoneInfo("America/New_York"))
    work = frame.copy()
    work["ts_et"] = work["ts"].dt.tz_convert(ZoneInfo("America/New_York"))

    if range_key == "1D":
        today = now_et.date()
        work = work[work["ts_et"].dt.date == today]

    return work.drop(columns=["ts_et"], errors="ignore")


def _frame_to_bars(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    bars: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        bars.append(
            {
                "ts": row.ts.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": None if pd.isna(row.volume) else float(row.volume),
            }
        )
    return bars


def _render_chart_card(ticker: str, series: dict[str, Any]) -> None:
    range_key = str(series.get("range_key", "3M"))
    point_count = int(series.get("point_count", 0))
    min_points = int(series.get("min_points", 0))
    flags = [str(x) for x in series.get("flags", [])]
    contract = series.get("contract", {}) if isinstance(series.get("contract"), dict) else {}

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**{ticker} {range_key}**")
    if range_key == "1D":
        st.caption("1D (RTH+Extended)")

    source = str(series.get("source", "unknown"))
    as_of = str(series.get("as_of", "unknown"))
    st.caption(f"as_of {as_of} · source={source} · points={point_count}")

    error = safe_get(series, "diagnostics.error")
    if source == "cache" and isinstance(error, str) and error:
        st.warning("Live chart fetch failed. Showing cached series when available.")
    if source == "none":
        st.info("No chart data available (live and cache empty).")

    if point_count < min_points:
        st.info(f"Insufficient data for {range_key} (got {point_count} points).")
    if flags:
        st.caption(f"quality_flags: {', '.join(flags)}")

    figure = _build_plotly_figure(series=series, title=f"{ticker} {range_key}")
    if figure is None:
        if point_count > 0:
            st.info("Plot rendering unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*keyword arguments have been deprecated.*Use `config` instead.*",
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"brain_chart_{ticker}_{range_key}",
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "responsive": True,
            },
        )

    if range_key == "Advanced" and contract:
        st.caption(
            f"Advanced source: {contract.get('source_range')} ({contract.get('period')} / {contract.get('interval')})"
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _build_plotly_figure(series: dict[str, Any], title: str):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

    frame = _bars_to_frame(series.get("bars", []))
    if frame.empty:
        return None

    advanced = str(series.get("range_key")) == "Advanced"
    fig = go.Figure()

    if advanced:
        fig.add_trace(
            go.Candlestick(
                x=frame["ts"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="Price",
                increasing_line_color="#24d07a",
                decreasing_line_color="#ef5350",
            )
        )

        if len(frame) >= 50:
            frame = frame.copy()
            frame["ema_50_viz"] = frame["close"].ewm(span=50, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=frame["ts"],
                    y=frame["ema_50_viz"],
                    mode="lines",
                    name="EMA50",
                    line={"color": "#00c853", "width": 1.2},
                )
            )
        if len(frame) >= 200:
            frame = frame.copy()
            frame["sma_200_viz"] = frame["close"].rolling(window=200, min_periods=200).mean()
            fig.add_trace(
                go.Scatter(
                    x=frame["ts"],
                    y=frame["sma_200_viz"],
                    mode="lines",
                    name="SMA200",
                    line={"color": "#ffb74d", "width": 1.2},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=frame["ts"],
                y=frame["close"],
                mode="lines",
                name="Close",
                line={"color": "#8ab4f8", "width": 2},
            )
        )

    fig.update_layout(
        title=title,
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
        paper_bgcolor="#101826",
        plot_bgcolor="#101826",
        font={"color": "#dce7f3", "size": 12},
        xaxis={"gridcolor": "rgba(255,255,255,0.04)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.04)", "side": "right"},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        height=320,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def _bars_to_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(bars)
    if frame.empty:
        return frame
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["ts", "open", "high", "low", "close"])
    return frame.sort_values("ts")


def _render_why_card(context_pack: dict, vm: dict) -> None:
    one_liner = vm.get("one_liner")
    drivers = vm.get("drivers", [])
    conflicts = vm.get("conflicts", [])
    watch = vm.get("watch", [])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**The Why**")

    if one_liner:
        st.markdown("**Consensus summary**")
        st.markdown(_normalize_hub_text(str(one_liner)))
        _render_hub_section("Drivers (Top 3)", drivers[:3])
        risk_items = conflicts[:3] if conflicts else watch[:3]
        _render_hub_section("Risks / Caveats", risk_items)
    else:
        fallback = build_why_fallback(context_pack=context_pack, vm=vm)
        st.markdown(f"**{fallback['title']}**")
        st.markdown(f"- Hub unavailable: {fallback['reason']}")
        st.markdown(f"- DRL action/confidence: {fallback['action_final']} / {fallback['confidence_cap']:.0f}")
        if fallback["drivers"]:
            st.markdown("- Top drivers:")
            for line in fallback["drivers"]:
                st.markdown(f"  - {line}")
        st.markdown(f"- Gates: {', '.join(fallback['gates']) if fallback['gates'] else 'None'}")
        st.markdown(f"- Conflicts: {', '.join(fallback['conflicts']) if fallback['conflicts'] else 'None'}")

    st.markdown("</div>", unsafe_allow_html=True)


def build_why_fallback(context_pack: dict, vm: dict) -> dict[str, Any]:
    drl_result = safe_get(context_pack, "drl.result", {}) or {}
    trace = safe_get(context_pack, "drl.result.decision_trace", {}) or {}

    return {
        "title": "The Why (Deterministic fallback)",
        "reason": _hub_missing_reason(context_pack),
        "action_final": str(drl_result.get("action_final", vm.get("drl_action_raw", "WAIT"))),
        "confidence_cap": float(drl_result.get("confidence_cap", vm.get("confidence_cap", 0.0)) or 0.0),
        "drivers": _trace_driver_lines(trace=trace, drl_result=drl_result),
        "gates": [str(x) for x in drl_result.get("gates_triggered", [])],
        "conflicts": [str(x) for x in drl_result.get("conflicts", [])],
    }


def _build_market_why_block(
    context_pack: dict,
    vm: dict[str, Any],
    primary_series: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_price = vm.get("last_price")
    canonical_source = safe_get(vm, "price_sanity.source", "none")
    drl_action = str(vm.get("drl_action_raw", "WAIT"))
    metrics = safe_get(context_pack, "indicators.metrics", {}) or {}
    hub_valid = bool(safe_get(context_pack, "meta.hub.hub_valid", False))
    one_liner = vm.get("one_liner")
    if one_liner and _should_render_hub_narrative(context_pack=context_pack, vm=vm, primary_series=primary_series):
        sanitized_one_liner = _sanitize_wait_language(text=str(one_liner), drl_action=drl_action)
        return {
            "mode": "HUB",
            "one_liner": sanitized_one_liner,
            "drivers": vm.get("drivers", []),
            "conflicts": vm.get("conflicts", []),
            "watch": vm.get("watch", []),
            "display_price": canonical_price,
            "display_price_source": canonical_source,
            "drl_action_raw": drl_action,
            "metrics": metrics,
            "hub_valid": hub_valid,
        }
    fallback = build_why_fallback(context_pack=context_pack, vm=vm)
    if one_liner:
        fallback = dict(fallback)
        fallback["reason"] = "Market data stale or unavailable; hub narrative suppressed."
    return {
        "mode": "FALLBACK",
        **fallback,
        "display_price": canonical_price,
        "display_price_source": canonical_source,
        "drl_action_raw": drl_action,
        "metrics": metrics,
        "hub_valid": hub_valid,
    }


def _should_render_hub_narrative(
    context_pack: dict,
    vm: dict[str, Any],
    primary_series: dict[str, Any] | None,
) -> bool:
    if bool(safe_get(context_pack, "meta.data_quality.overall_stale", False)):
        return False

    last_price = vm.get("last_price")
    if not isinstance(last_price, (int, float)):
        return False

    quote_source = str(safe_get(vm, "quote.source", "none")).strip().lower()
    if quote_source in {"", "none"}:
        return False

    series = primary_series if isinstance(primary_series, dict) else {}
    series_source = str(series.get("source", "none")).strip().lower()
    if series_source in {"", "none"}:
        return False

    blocking_flags = {"MISSING", "EMPTY_LIVE", "EMPTY_CACHE"}
    series_flags = {str(flag).upper() for flag in (series.get("flags", []) or [])}
    if series_flags & blocking_flags:
        return False

    return True


def _hub_missing_reason(context_pack: dict) -> str:
    hub_reason = str(safe_get(context_pack, "meta.hub.reason", "") or "").strip()
    if hub_reason:
        if hub_reason == "LLM not configured":
            return "LLM not configured (run make llm-smoke; ensure .env loaded)."
        match = re.search(r"HUB_FORBIDDEN_TERM:([A-Za-z_]+)", hub_reason)
        if match:
            return f"Hub validation failed (forbidden term: {match.group(1).lower()})"
        return hub_reason

    notes = [str(n).upper() for n in (safe_get(context_pack, "meta.data_quality.notes", []) or [])]

    if not _is_llm_configured():
        return "LLM not configured (run make llm-smoke; ensure .env loaded)."
    if any("HUB_VALIDATION_FAILED" in note for note in notes):
        return "Hub validation failed"
    if any("BEDROCK_UNAVAILABLE" in note or "HUB_DISABLED" in note for note in notes):
        return "LLM reachable but hub generation disabled"
    return "Context pack missing hub artifact"


def _is_llm_configured() -> bool:
    return bool(hub_is_llm_configured())


def _attempt_sync_why_generation(
    *,
    ticker: str,
    why_signature: str,
    range_key: str,
    context_pack: dict[str, Any],
    quote: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    symbol = str(ticker or "").strip().upper()
    signature = str(why_signature or "").strip().lower()
    if not symbol or not signature:
        return {
            "status": "skipped",
            "error": "missing_signature",
            "elapsed_ms": 0.0,
            "artifact": None,
            "llm_usage": {},
        }

    throttle_state = st.session_state.setdefault("why_sync_attempt_at", {})
    if not isinstance(throttle_state, dict):
        throttle_state = {}
        st.session_state["why_sync_attempt_at"] = throttle_state
    throttle_key = f"{symbol}:{signature}"
    now_ts = time.time()
    last_ts_raw = throttle_state.get(throttle_key, 0.0)
    try:
        last_ts = float(last_ts_raw)
    except (TypeError, ValueError):
        last_ts = 0.0
    if (now_ts - last_ts) < 90.0:
        return {
            "status": "throttled",
            "error": "",
            "elapsed_ms": 0.0,
            "artifact": None,
            "llm_usage": {},
        }
    throttle_state[throttle_key] = now_ts

    started = time.perf_counter()
    try:
        cfg = resolve_bedrock_config()
        hub_result = generate_hub_for_context_pack(
            context_pack=context_pack,
            now_iso=now_iso(),
            bedrock_config=cfg,
            request_timeout_seconds=float(timeout_seconds),
        )
        response = {
            "status": str(hub_result.status),
            "mode": str(hub_result.mode),
            "reason": hub_result.reason,
            "hub_card": hub_result.hub_card,
            "hub_valid": bool(hub_result.hub_valid),
            "from_cache": bool(hub_result.from_cache),
            "llm_usage": hub_result.llm_usage if isinstance(hub_result.llm_usage, dict) else {},
        }
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000.0, 2)
        err = str(exc)[:180]
        status = "timeout" if "timed out" in err.lower() or "timeout" in err.lower() else "error"
        return {
            "status": status,
            "error": err,
            "elapsed_ms": elapsed,
            "artifact": None,
            "llm_usage": {},
        }

    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    hub_card = response.get("hub_card")
    status = str(response.get("status", "error"))
    reason = str(response.get("reason", "") or "")
    llm_usage = response.get("llm_usage", {}) if isinstance(response.get("llm_usage"), dict) else {}
    if status != "present" or not isinstance(hub_card, dict):
        return {
            "status": "error",
            "error": reason or f"hub_status:{status}",
            "elapsed_ms": elapsed,
            "artifact": None,
            "llm_usage": llm_usage,
        }

    generated_at = now_iso()
    hub_meta = {
        "status": "present",
        "mode": str(response.get("mode", "NORMAL")),
        "reason": None,
        "hub_valid": bool(response.get("hub_valid", True)),
        "from_cache": bool(response.get("from_cache", False)),
        "llm_usage": llm_usage,
    }
    save_why_artifact(
        signature=signature,
        ticker=symbol,
        hub_card=hub_card,
        hub_meta=hub_meta,
        generated_at=generated_at,
    )
    resolved_signature = build_why_signature(
        ticker=symbol,
        drl_result=safe_get(context_pack, "drl.result", {}) or {},
        indicators=safe_get(context_pack, "indicators", {}) or {},
        quote=quote if isinstance(quote, dict) else {},
        range_key=str(range_key or "1D"),
    )
    resolved_sig = str(resolved_signature or "").strip().lower()
    if resolved_sig and resolved_sig != signature:
        save_why_artifact(
            signature=resolved_sig,
            ticker=symbol,
            hub_card=hub_card,
            hub_meta=hub_meta,
            generated_at=generated_at,
        )
    artifact = WhyArtifact(
        signature=signature,
        ticker=symbol,
        generated_at=generated_at,
        hub_card=hub_card,
        hub_meta=hub_meta,
        source="live_sync",
    )
    return {
        "status": "success",
        "error": "",
        "elapsed_ms": elapsed,
        "artifact": artifact,
        "llm_usage": llm_usage,
    }


def _enqueue_why_refresh(*, ticker: str, why_signature: str, range_key: str) -> bool:
    symbol = str(ticker or "").strip().upper()
    signature = str(why_signature or "").strip().lower()
    if not symbol or not signature:
        return False

    queue_state = st.session_state.setdefault("why_refresh_queue_at", {})
    if not isinstance(queue_state, dict):
        queue_state = {}
        st.session_state["why_refresh_queue_at"] = queue_state

    throttle_key = f"{symbol}:{signature}"
    now_ts = time.time()
    last_ts_raw = queue_state.get(throttle_key, 0.0)
    try:
        last_ts = float(last_ts_raw)
    except (TypeError, ValueError):
        last_ts = 0.0
    if (now_ts - last_ts) < 90.0:
        return False

    enqueue_prewarm_request(
        scope="brain",
        tickers={symbol},
        range_keys=(str(range_key or "1D").strip().upper() or "1D",),
        reason="why_refresh",
        requested_by="ui",
        metadata={"why_signature": signature},
    )
    queue_state[throttle_key] = now_ts
    return True


def _set_hub_refresh_reason(*, context_pack: dict[str, Any], reason: str) -> None:
    if not isinstance(context_pack, dict):
        return
    meta = context_pack.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        context_pack["meta"] = meta
    hub = meta.setdefault("hub", {})
    if not isinstance(hub, dict):
        hub = {}
        meta["hub"] = hub
    hub["status"] = str(hub.get("status", "missing") or "missing")
    hub["mode"] = str(hub.get("mode", "DEGRADED") or "DEGRADED")
    hub["hub_valid"] = bool(hub.get("hub_valid", False))
    hub["reason"] = str(reason)


def _attach_why_meta(
    *,
    context_pack: dict[str, Any],
    why_signature: str | None,
    why_cache_state: str,
    why_sync_status: str = "not_attempted",
    why_sync_error: str = "",
    why_sync_elapsed_ms: float = 0.0,
    why_llm_usage: dict[str, Any] | None = None,
) -> None:
    if not isinstance(context_pack, dict):
        return
    meta = context_pack.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        context_pack["meta"] = meta
    hub = meta.setdefault("hub", {})
    if not isinstance(hub, dict):
        hub = {}
        meta["hub"] = hub
    requested = str(why_signature or "").strip().lower()
    if requested:
        hub["why_signature_requested"] = requested
    loaded = str(hub.get("why_signature", "") or "").strip().lower()
    if not loaded and requested:
        loaded = requested
        hub["why_signature"] = loaded
    hub["why_signature_loaded"] = loaded or "none"
    hub["why_source"] = str(why_cache_state or "none")
    hub["why_sync_status"] = str(why_sync_status or "not_attempted")
    hub["why_sync_error"] = str(why_sync_error or "")
    hub["why_sync_elapsed_ms"] = round(max(0.0, float(why_sync_elapsed_ms or 0.0)), 2)
    usage = why_llm_usage if isinstance(why_llm_usage, dict) else hub.get("llm_usage")
    hub["llm_usage"] = usage if isinstance(usage, dict) else {}


def _trace_driver_lines(trace: dict[str, Any], drl_result: dict[str, Any]) -> list[str]:
    lines: list[tuple[float, str]] = []
    components = trace.get("score_components", {})
    if isinstance(components, dict):
        for name, details in components.items():
            if not isinstance(details, dict):
                continue
            score = _to_float(details.get("score"), default=0.0)
            lines.append((abs(score), f"{name}: {score:+.2f}"))

    lines.sort(key=lambda item: item[0], reverse=True)
    top = [text for _, text in lines[:3] if text]
    if top:
        return top

    return [
        f"regime_1D: {drl_result.get('regime_1D', 'NEUTRAL')}",
        f"regime_1W: {drl_result.get('regime_1W', 'NEUTRAL')}",
        f"score_final: {trace.get('score_final', 'n/a')}",
    ]


def _render_hub_section(title: str, items: list[dict]) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.markdown("<div class='tiny muted'>None</div>", unsafe_allow_html=True)
        return
    for item in items:
        text = _normalize_hub_text(str(item.get("text", "")).strip())
        st.markdown(f"- {text}")


def _collect_hub_citations(items: list[dict]) -> list[str]:
    citations: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for citation in item.get("citations", []):
            cid = str(citation)
            if cid and cid not in citations:
                citations.append(cid)
    return citations


def _render_content_references(refs: list[Any]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        raw_value = str(ref).strip()
        if not raw_value:
            continue
        parts = [part.strip() for part in raw_value.split(",")]
        for part in parts:
            normalized = _strip_ref_prefixes(part)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
    if not ordered:
        return "none"
    labels = [html.escape(_citation_label_from_id(ref)) for ref in ordered]
    return ", ".join(labels)


def _citation_label_from_id(ref_id: str) -> str:
    normalized = _strip_ref_prefixes(ref_id)
    if normalized.startswith("indicator:"):
        return normalized.split(":", 1)[1]
    return normalized


def _strip_ref_prefixes(text: str) -> str:
    cleaned = str(text).strip()
    while True:
        updated = cleaned
        updated = re.sub(r"(?i)^\s*refs\s*:\s*", "", updated).strip()
        updated = re.sub(r"(?i)^\s*citations\s*:\s*", "", updated).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _normalize_hub_text(text: str) -> str:
    normalized = text.replace("\\n", "\n")
    parts = [segment.strip() for segment in normalized.splitlines() if segment.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts)


def _render_evidence(
    context_pack: dict,
    series_by_range: dict[str, dict[str, Any]],
    vm: dict[str, Any] | None = None,
) -> None:
    with st.expander("View Evidence", expanded=False):
        range_key = st.selectbox("Price series", options=RANGE_ORDER, index=0, key="brain_evidence_range")
        series = series_by_range.get(range_key, {"range_key": range_key, "bars": [], "flags": [], "point_count": 0})
        series_as_of_local, _ = _format_as_of_local_utc(series.get("as_of"))
        metrics = safe_get(context_pack, "indicators.metrics", {}) or {}
        indicator_keys = ["price_last", "ema_50", "sma_200", "rsi_14", "macd", "macd_signal", "adx_14", "vroc_14", "atr_pct"]
        indicator_rows = [{"key": key, "value": metrics.get(key)} for key in indicator_keys if key in metrics]
        evidence_payload = {
            "range_key": range_key,
            "points": int(series.get("point_count", 0)),
            "as_of": series_as_of_local,
            "source": str(series.get("source", "unknown")),
            "quality_flags": list(series.get("flags", [])),
            "indicator_ts": safe_get(context_pack, "indicators.as_of"),
            "latest_quote": safe_get(context_pack, "meta.latest_quote", {}),
            "drl_action": safe_get(context_pack, "drl.result.action_final", "WAIT"),
            "confidence_cap": safe_get(context_pack, "drl.result.confidence_cap", 0),
            "why_signature_requested": safe_get(context_pack, "meta.hub.why_signature_requested", "none"),
            "why_signature_loaded": safe_get(context_pack, "meta.hub.why_signature_loaded", safe_get(context_pack, "meta.hub.why_signature", "none")),
            "why_source": safe_get(context_pack, "meta.hub.why_source", "none"),
        }
        summary_rows = [
            {"k": "range_key", "v": evidence_payload["range_key"]},
            {"k": "points", "v": evidence_payload["points"]},
            {"k": "as_of", "v": evidence_payload["as_of"]},
            {"k": "source", "v": evidence_payload["source"]},
            {"k": "quality_flags", "v": ", ".join(evidence_payload["quality_flags"]) or "none"},
            {"k": "indicator_ts", "v": evidence_payload["indicator_ts"]},
            {"k": "drl_action", "v": evidence_payload["drl_action"]},
            {"k": "confidence_cap", "v": evidence_payload["confidence_cap"]},
            {"k": "why_signature_requested", "v": evidence_payload["why_signature_requested"]},
            {"k": "why_signature_loaded", "v": evidence_payload["why_signature_loaded"]},
            {"k": "why_source", "v": evidence_payload["why_source"]},
        ]
        st.dataframe(df_for_streamlit(pd.DataFrame(summary_rows)), width="stretch", hide_index=True)
        st.download_button(
            "Download evidence (JSON)",
            data=to_json_bytes({"summary": evidence_payload, "indicators": indicator_rows}),
            file_name=f"evidence-{safe_get(context_pack, 'meta.ticker', 'ticker')}.json",
            mime="application/json",
            width="stretch",
            key="brain_evidence_download_json",
        )
        st.download_button(
            "Download indicators (CSV)",
            data=rows_to_csv_bytes(indicator_rows, fieldnames=["key", "value"]),
            file_name=f"indicators-{safe_get(context_pack, 'meta.ticker', 'ticker')}.csv",
            mime="text/csv",
            width="stretch",
            key="brain_evidence_download_csv",
        )


def _citation_labels(context_pack: dict) -> dict[str, str]:
    labels: dict[str, str] = {}

    metrics = safe_get(context_pack, "indicators.metrics", {}) or {}
    for key, value in metrics.items():
        labels[f"indicator:{key}"] = f"{key}={value}"

    for item in _extract_channel_items(context_pack, key="news"):
        news_id = str(item.get("id", "")).strip()
        if news_id:
            labels[f"news:{news_id}"] = str(item.get("title", "news item"))

    for item in _extract_channel_items(context_pack, key="macro"):
        macro_id = str(item.get("id", "")).strip()
        if macro_id:
            labels[f"macro:{macro_id}"] = str(item.get("label", "macro item"))

    return labels


def _render_diagnostics(
    selected_ticker: str,
    context_pack: dict,
    policy_path: str,
    series_by_range: dict[str, dict[str, Any]],
    vm: dict[str, Any],
    hydration_diag: dict[str, Any] | None = None,
) -> None:
    with st.expander("Diagnostics", expanded=False):
        quote_source = str(safe_get(vm, "quote.source", "none")).lower()
        market_ok = int(series_by_range.get("1D", {}).get("point_count", 0)) > 0 or quote_source != "none"
        news_items = _extract_channel_items(context_pack, key="news")
        macro_items = _extract_channel_items(context_pack, key="macro")
        diagnostics_payload = {
            "market_data_provider": "UP" if market_ok else "DOWN",
            "quote_source": quote_source or "none",
            "quote_session_state": safe_get(vm, "quote.session_state", "unknown") or "unknown",
            "quote_extended_label": safe_get(vm, "quote.extended_label", "none") or "none",
            "quote_show_extended_session": bool(safe_get(vm, "quote.show_extended_session", False)),
            "news_tool": "UP" if news_items else "NO_DATA",
            "macro_tool": "UP" if macro_items else "NO_DATA",
            "hub_status": safe_get(context_pack, "meta.hub.status", "missing"),
            "hub_mode": safe_get(context_pack, "meta.hub.mode", "DEGRADED"),
            "hub_valid": bool(safe_get(context_pack, "meta.hub.hub_valid", False)),
            "hub_reason": safe_get(context_pack, "meta.hub.reason", "none") or "none",
            "why_signature_requested": safe_get(context_pack, "meta.hub.why_signature_requested", "none") or "none",
            "why_signature_loaded": safe_get(context_pack, "meta.hub.why_signature_loaded", safe_get(context_pack, "meta.hub.why_signature", "none")) or "none",
            "why_source": safe_get(context_pack, "meta.hub.why_source", "none") or "none",
            "why_sync_status": safe_get(context_pack, "meta.hub.why_sync_status", "not_attempted") or "not_attempted",
            "why_sync_elapsed_ms": round(float(safe_get(context_pack, "meta.hub.why_sync_elapsed_ms", 0.0) or 0.0), 2),
            "why_sync_error": safe_get(context_pack, "meta.hub.why_sync_error", "") or "",
            "bars_cache_hits": sum(1 for s in series_by_range.values() if s.get("diagnostics", {}).get("cache_hit")),
            "bars_cache_misses": sum(1 for s in series_by_range.values() if not s.get("diagnostics", {}).get("cache_hit")),
            "invariants_ok": bool(st.session_state.get("invariants_last_result", {}).get("ok", True)),
        }
        llm_usage = safe_get(context_pack, "meta.hub.llm_usage", {}) or {}
        diagnostics_payload["why_llm_input_tokens"] = int(safe_get(llm_usage, "input_tokens", 0) or 0)
        diagnostics_payload["why_llm_output_tokens"] = int(safe_get(llm_usage, "output_tokens", 0) or 0)
        diagnostics_payload["why_llm_total_tokens"] = int(safe_get(llm_usage, "total_tokens", 0) or 0)
        diagnostics_payload["why_llm_latency_ms"] = round(float(safe_get(llm_usage, "latency_ms", 0.0) or 0.0), 2)
        hydration = hydration_diag if isinstance(hydration_diag, dict) else {}
        diagnostics_payload["brain_hydration_phase"] = str(hydration.get("phase", "ready"))
        diagnostics_payload["brain_stage1_ms"] = round(float(hydration.get("stage1_ms", 0.0) or 0.0), 2)
        diagnostics_payload["brain_stage2_ms"] = round(float(hydration.get("stage2_ms", 0.0) or 0.0), 2)

        prewarm_status = load_prewarm_status()
        diagnostics_payload["prewarm_last_completed_at"] = str(prewarm_status.get("last_completed_at", "n/a"))
        diagnostics_payload["prewarm_last_reason"] = str(prewarm_status.get("last_reason", "n/a"))
        diagnostics_payload["prewarm_queue_depth"] = int(prewarm_queue_depth())

        cache_snapshot = cache_hygiene_snapshot()
        diagnostics_payload["cache_file_count"] = int(cache_snapshot.get("file_count", 0))
        diagnostics_payload["cache_total_mb"] = float(cache_snapshot.get("total_size_mb", 0.0))
        diagnostics_payload["cache_empty_files"] = int(cache_snapshot.get("empty_files", 0))
        refresh_report = st.session_state.get("last_refresh_report", {})
        if isinstance(refresh_report, dict):
            diagnostics_payload["refresh_scope"] = refresh_report.get("scope", "none")
            diagnostics_payload["refresh_tickers"] = refresh_report.get("tickers", 0)
            diagnostics_payload["refresh_ranges"] = ",".join(str(x) for x in (refresh_report.get("ranges", []) or []))
            diagnostics_payload["refresh_attempted"] = refresh_report.get("attempted", 0)
            diagnostics_payload["refresh_live"] = refresh_report.get("live", 0)
            diagnostics_payload["refresh_cache"] = refresh_report.get("cache", 0)
            diagnostics_payload["refresh_none"] = refresh_report.get("none", 0)
            diagnostics_payload["refresh_errors"] = refresh_report.get("errors", 0)
            hygiene = refresh_report.get("cache_hygiene", {})
            if isinstance(hygiene, dict):
                diagnostics_payload["cache_hygiene_scanned"] = hygiene.get("scanned", 0)
                diagnostics_payload["cache_hygiene_removed"] = hygiene.get("removed", 0)
        llm_smoke = _load_llm_smoke_status()
        diagnostics_payload["llm_smoke"] = (
            f"{llm_smoke.get('status', 'unknown')} @ {llm_smoke.get('checked_at', 'n/a')}"
            if llm_smoke
            else "unknown"
        )
        diag_rows = [{"k": key, "v": value} for key, value in diagnostics_payload.items()]
        st.dataframe(df_for_streamlit(pd.DataFrame(diag_rows)), width="stretch", hide_index=True)
        st.download_button(
            "Download diagnostics (JSON)",
            data=to_json_bytes(diagnostics_payload),
            file_name=f"diagnostics-{selected_ticker}.json",
            mime="application/json",
            width="stretch",
            key="brain_diagnostics_download_json",
        )

        st.markdown("---")
        _render_invariants_quickcheck(policy_path=policy_path)
        st.markdown("---")
        render_replay_tools(selected_ticker=selected_ticker, context_pack=context_pack, policy_path=policy_path)


def _render_invariants_quickcheck(policy_path: str) -> None:
    if "invariants_last_result" not in st.session_state:
        st.session_state["invariants_last_result"] = run_invariants_quickcheck(policy_path)

    if st.button("Run Invariants Quickcheck", key="brain_invariants_quickcheck_button"):
        st.session_state["invariants_last_result"] = run_invariants_quickcheck(policy_path)

    result = st.session_state.get("invariants_last_result", {})
    timestamp = result.get("timestamp", "unknown")

    st.markdown("**Invariants**")
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


def _load_llm_smoke_status() -> dict[str, Any] | None:
    path = Path(".cache/llm_smoke_status.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _last_close(bars: list[dict]) -> float:
    if not bars:
        return 0.0
    try:
        return float(bars[-1].get("close", 0.0))
    except Exception:
        return 0.0


def _series_last_close(series: dict[str, Any] | None) -> float | None:
    if not isinstance(series, dict):
        return None
    bars = series.get("bars", [])
    if not isinstance(bars, list) or not bars:
        return None
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    try:
        return float(last.get("close"))
    except (TypeError, ValueError):
        return None


def _first_non_empty_close(
    fallback_series: dict[str, dict[str, Any]],
    order: list[str],
) -> float | None:
    for range_key in order:
        close_value = _series_last_close(fallback_series.get(range_key))
        if close_value is not None:
            return close_value
    return None


def _extract_channel_items(context_pack: dict, key: str) -> list[dict]:
    direct = context_pack.get(key)
    if isinstance(direct, dict):
        items = direct.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    mcp = context_pack.get("mcp", {})
    if isinstance(mcp, dict):
        channel = mcp.get(key)
        if isinstance(channel, dict):
            items = channel.get("items", [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_display_price_info(series: dict[str, Any], metrics: dict[str, Any], vm: dict[str, Any]) -> dict[str, Any]:
    bars = series.get("bars", []) if isinstance(series, dict) else []
    indicator_price = metrics.get("price_last")
    flags: list[str] = []
    quote_close = safe_get(vm, "quote.close_price", None)
    quote_after = safe_get(vm, "quote.after_hours_price", None)
    source = str(safe_get(vm, "price_sanity.source", "none"))
    display_price = safe_get(vm, "price_sanity.display_price", None)
    if not isinstance(display_price, (int, float)):
        display_price = None
    flags = [str(x) for x in (safe_get(vm, "price_sanity.quality_flags", []) or [])]

    bars_close = None
    if isinstance(bars, list) and bars:
        try:
            bars_close = float(bars[-1].get("close"))
        except (TypeError, ValueError):
            bars_close = None

    return {
        "display_price": display_price if display_price is not None else "N/A",
        "source": source,
        "flags": _unique(flags),
        "bars_last_close": bars_close if bars_close is not None else "N/A",
        "indicator_price_last": indicator_price if indicator_price is not None else "N/A",
        "quote_close": quote_close if isinstance(quote_close, (int, float)) else "N/A",
        "quote_after_hours": quote_after if isinstance(quote_after, (int, float)) else "N/A",
        "quote_latest": safe_get(vm, "quote.latest_price", "N/A"),
        "quote_latest_ts": safe_get(vm, "quote.latest_ts", "N/A"),
        "quote_close_ts": safe_get(vm, "quote.close_ts", "N/A"),
        "quote_after_hours_ts": safe_get(vm, "quote.after_hours_ts", "N/A"),
    }


def _sanitize_wait_language(text: str, drl_action: str) -> str:
    normalized = str(text)
    if str(drl_action).upper() != "WAIT":
        return normalized
    lowered = normalized.lower()
    if any(token in lowered for token in ["strong buy", "buy now", "strong sell", "sell now"]):
        return "Current setup supports WAIT while directional evidence remains mixed."
    return normalized


def _is_provider_failure(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    if not lowered:
        return False
    tokens = [
        "provider_down",
        "fetch_failed",
        "yfinance_unavailable",
        "network",
        "connection",
        "timeout",
        "http",
        "dns",
    ]
    return any(token in lowered for token in tokens)


def _format_as_of_local_utc(value: Any) -> tuple[str, str]:
    if value is None:
        return "unknown", "unknown"
    text = str(value).strip()
    if not text:
        return "unknown", "unknown"
    try:
        dt = parse_iso(text)
    except Exception:
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = parse_iso(text)
        except Exception:
            return str(value), str(value)
    local_dt = dt.astimezone()
    utc_dt = dt.astimezone(timezone.utc)
    return local_dt.isoformat(timespec="minutes"), utc_dt.isoformat(timespec="minutes")


def _format_change_line(abs_value: Any, pct_value: Any, *, label: str) -> str:
    try:
        abs_f = float(abs_value)
        pct_f = float(pct_value)
    except (TypeError, ValueError):
        return f"{label}: —"
    sign = "▲" if abs_f >= 0 else "▼"
    sign_money = "+" if abs_f >= 0 else ""
    sign_pct = "+" if pct_f >= 0 else ""
    return f"{sign} {sign_money}{format_money(abs_f)} ({sign_pct}{pct_f:.2f}%) {label}"
