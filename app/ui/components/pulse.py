from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from app.core.marketdata.chart_fetcher import ChartFetcher
from app.core.marketdata.query_graph import MarketQueryService
from app.core.query.contracts import run_short_query
from app.core.marketdata.yfinance_provider import YahooChartMarketDataProvider
from app.core.marketdata.quotes import get_quote_snapshot, quote_snapshot_to_dict
from app.core.portfolio.portfolio_store import remove_holding, upsert_holding
from app.ui.components.pulse_badges import (
    TOOLTIP_AGE,
    TOOLTIP_DEGRADED,
    TOOLTIP_TOOL_DOWN,
    compute_pulse_badges,
)
from app.ui.components.ui_utils import format_money, format_pct, safe_get
from app.ui.viewmodels.pulse_vm import build_pulse_row_vm

_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
_LOCAL_UNIVERSE_FALLBACK = {"AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG"}


def render_pulse(
    portfolio: Any,
    context_loader: Callable[..., dict],
    market_query: MarketQueryService | None = None,
) -> None:
    st.markdown("<div class='card'><div class='section-title'>The Pulse</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='pulse-block-gap'></div>", unsafe_allow_html=True)
    _render_manage_holdings(portfolio)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    holdings = list(getattr(portfolio, "holdings", []) or [])
    if not holdings:
        st.info("No holdings yet. Add one from Manage Holdings.")
        return

    for holding in holdings:
        _render_holding_card(holding=holding, context_loader=context_loader, market_query=market_query)


def _render_manage_holdings(portfolio: Any) -> None:
    st.session_state.setdefault("pulse_manage_open", False)
    st.session_state.setdefault("pulse_form_nonce", 0)

    holdings = list(getattr(portfolio, "holdings", []) or [])
    existing = {str(getattr(h, "ticker", "")).upper(): h for h in holdings}
    nonce = int(st.session_state.get("pulse_form_nonce", 0))
    ticker_key = f"pulse_manage_ticker_{nonce}"
    avg_key = f"pulse_manage_avg_cost_{nonce}"
    qty_key = f"pulse_manage_quantity_{nonce}"

    with st.container(border=True):
        head_left, head_right = st.columns([4.5, 1.2], vertical_alignment="center")
        with head_left:
            st.markdown("**Manage Holdings**")
        with head_right:
            if st.button("+ Add", key="pulse_manage_toggle", help="Add / Update holding", width="stretch"):
                st.session_state["pulse_manage_open"] = not bool(st.session_state.get("pulse_manage_open", False))

        if bool(st.session_state.get("pulse_manage_open", False)):
            st.markdown("<div class='tiny muted'>Add new holding.</div>", unsafe_allow_html=True)
            ticker_raw = st.text_input("Ticker", key=ticker_key, placeholder="AAPL")
            avg_raw = st.text_input("Avg Cost", key=avg_key, placeholder="180.55")
            qty_raw = st.text_input("Quantity", key=qty_key, placeholder="10")

            ticker_norm = _normalize_ticker(ticker_raw)
            submit_label = "Update" if ticker_norm and ticker_norm in existing else "Add"
            submitted = st.button(submit_label, key=f"pulse_manage_submit_{nonce}", width="stretch")

            if submitted:
                error = _validate_manage_inputs(ticker_raw=ticker_raw, avg_raw=avg_raw, qty_raw=qty_raw)
                if error:
                    st.warning(error)
                else:
                    is_new = bool(ticker_norm and ticker_norm not in existing)
                    if is_new:
                        exists, verify_msg = _validate_ticker_exists(str(ticker_norm))
                        if not exists:
                            st.warning(verify_msg)
                            return
                    avg_cost = float(avg_raw)
                    quantity = float(qty_raw)
                    upsert_holding(ticker=ticker_norm or "", avg_cost=avg_cost, quantity=quantity)
                    st.session_state["selected_ticker"] = ticker_norm
                    st.session_state["pulse_form_nonce"] = nonce + 1
                    st.session_state["pulse_manage_open"] = False
                    st.rerun()


def _render_holding_card(
    holding: Any,
    context_loader: Callable[..., dict],
    market_query: MarketQueryService | None = None,
) -> None:
    ticker = str(_holding_value(holding, "ticker") or "").strip().upper()
    if not ticker:
        return

    query = market_query or MarketQueryService(cache_dir=".cache/charts", context_loader=context_loader)
    short_result = run_short_query(ticker=ticker, market_query=query)
    main_pack = short_result.context_pack
    if not isinstance(main_pack, dict):
        main_pack = _fallback_context_pack(ticker=ticker)
    quote = short_result.quote
    if not isinstance(quote, dict):
        quote = {}
    if str(quote.get("source", "none")).lower() in {"", "none"}:
        fallback_quote = safe_get(main_pack, "meta.latest_quote", {})
        if isinstance(fallback_quote, dict):
            quote = dict(fallback_quote)
    primary_chart_series = _chart_series_to_dict(short_result.series_1d)
    primary_chart_bars = primary_chart_series.get("bars", []) if isinstance(primary_chart_series, dict) else []
    sparkline_source = _downsample_bars(primary_chart_bars, max_points=40)
    sparkline_points = _sparkline_points_from_bars(sparkline_source, n=40)

    vm = build_pulse_row_vm(
        holding=_holding_to_dict(holding),
        context_pack=main_pack,
        quote=quote,
        primary_series_close=_last_close_or_none(primary_chart_bars),
    )

    quantity = float(vm.get("quantity", 0.0) or 0.0)
    avg_cost = float(vm.get("avg_cost", 0.0) or 0.0)
    last_price = vm.get("last_price")
    last_price_val = float(last_price) if isinstance(last_price, (int, float)) else None

    market_value = (quantity * last_price_val) if last_price_val is not None else None
    total_return_dollars = ((last_price_val - avg_cost) * quantity) if last_price_val is not None else None
    total_return_pct = (
        ((last_price_val - avg_cost) / avg_cost * 100.0)
        if (last_price_val is not None and avg_cost > 0)
        else None
    )

    today_change_per_share = _to_optional_float(vm.get("today_abs"))
    today_change_pct = _to_optional_float(vm.get("today_pct"))
    today_return_dollars = (
        today_change_per_share * quantity
        if isinstance(today_change_per_share, (int, float))
        else None
    )

    badge_state = compute_pulse_badges(main_pack)

    price_flags = [str(x) for x in safe_get(vm, "price_sanity.quality_flags", []) or []]
    if price_flags:
        badge_state = dict(badge_state)
        reasons = [str(x) for x in badge_state.get("reasons", [])]
        for flag in price_flags:
            if flag not in reasons:
                reasons.append(flag)
        badge_state["reasons"] = reasons
        if "MISSING_BARS" in price_flags or "PRICE_MISMATCH" in price_flags:
            badge_state["show_degraded"] = True

    if not primary_chart_bars:
        badge_state = dict(badge_state)
        reasons = [str(x) for x in badge_state.get("reasons", [])]
        if "MISSING_BARS" not in reasons:
            reasons.append("MISSING_BARS")
        badge_state["reasons"] = reasons
        badge_state["show_degraded"] = True

    quote_source = str(safe_get(vm, "quote.source", "none")).lower()
    chart_source = str(primary_chart_series.get("source", "none")).lower()
    if quote_source == "none" and chart_source == "none":
        badge_state = dict(badge_state)
        reasons = [str(x) for x in badge_state.get("reasons", [])]
        if "PRICE_TOOL_DOWN" not in reasons:
            reasons.append("PRICE_TOOL_DOWN")
        badge_state["reasons"] = reasons
        badge_state["show_tool_down"] = True
        badge_state["show_degraded"] = True

    action_ui = str(vm.get("ui_action_label", "HOLD"))
    action_raw = str(vm.get("drl_action_raw", "WAIT"))
    confidence = float(vm.get("confidence_cap", 0.0) or 0.0)
    pill_class = _pulse_pill_class(action_raw)
    pill_arrow = _pill_arrow(action_ui)

    company = _company_name(ticker)
    sparkline_svg = _build_sparkline_svg(sparkline_points=sparkline_points, action=action_raw)
    today_sign = "+" if isinstance(today_change_per_share, (int, float)) and today_change_per_share > 0 else ""
    today_return_sign = "+" if isinstance(today_return_dollars, (int, float)) and today_return_dollars > 0 else ""
    total_sign = "+" if isinstance(total_return_dollars, (int, float)) and total_return_dollars > 0 else ""
    today_dir = "▲" if isinstance(today_change_per_share, (int, float)) and today_change_per_share >= 0 else "▼"
    price_delta_text = (
        f"{today_sign}{format_money(today_change_per_share)} ({format_pct(today_change_pct)})"
        if isinstance(today_change_per_share, (int, float)) and isinstance(today_change_pct, (int, float))
        else "—"
    )
    price_text = _money_or_dash(last_price_val)
    market_text = _money_or_dash(market_value)
    today_line_html = "<div class='quote-line neu'>Today: —</div>"
    if isinstance(today_change_per_share, (int, float)) and isinstance(today_change_pct, (int, float)):
        today_line_class = "pos" if today_change_per_share > 0 else "neg" if today_change_per_share < 0 else "neu"
        today_line_html = (
            f"<div class='quote-line {today_line_class}'>"
            f"{today_dir} {html.escape(price_delta_text)} Today"
            "</div>"
        )
    ah_change_abs = _to_optional_float(safe_get(vm, "quote.after_hours_change_abs", None))
    ah_change_pct = _to_optional_float(safe_get(vm, "quote.after_hours_change_pct", None))
    show_extended = bool(safe_get(vm, "quote.show_extended_session", False))
    extended_label = str(safe_get(vm, "quote.extended_label", "") or "").strip() or "After-hours"
    extended_icon = "🌙" if extended_label == "After-hours" else "🌅"
    ah_line_class = "neu"
    ah_line_html = ""
    if isinstance(ah_change_abs, (int, float)) and isinstance(ah_change_pct, (int, float)):
        ah_dir = "▲" if ah_change_abs >= 0 else "▼"
        ah_sign = "+" if ah_change_abs >= 0 else ""
        ah_change_line = (
            f"{ah_dir} {ah_sign}{format_money(ah_change_abs)} "
            f"({format_pct(ah_change_pct)}) {extended_icon} {extended_label}"
        )
        ah_line_class = "pos" if ah_change_abs > 0 else "neg" if ah_change_abs < 0 else "neu"
        if show_extended:
            ah_line_html = f"<div class='quote-line {ah_line_class}'>{html.escape(ah_change_line)}</div>"
    else:
        ah_change_line = f"{extended_icon} {extended_label}: —"
        if show_extended:
            ah_line_html = f"<div class='quote-line neu'>{html.escape(ah_change_line)}</div>"

    today_return_pct_text = (
        f"{today_return_sign}{format_money(today_return_dollars)} ({format_pct(today_change_pct)})"
        if isinstance(today_return_dollars, (int, float)) and isinstance(today_change_pct, (int, float))
        else "--"
    )
    total_return_text = (
        f"{total_sign}{format_money(total_return_dollars)} ({format_pct(total_return_pct)})"
        if isinstance(total_return_dollars, (int, float)) and isinstance(total_return_pct, (int, float))
        else "--"
    )
    today_return_class = (
        "pos"
        if isinstance(today_return_dollars, (int, float)) and today_return_dollars > 0
        else "neg"
        if isinstance(today_return_dollars, (int, float)) and today_return_dollars < 0
        else "neu"
    )
    total_return_class = (
        "pos"
        if isinstance(total_return_dollars, (int, float)) and total_return_dollars > 0
        else "neg"
        if isinstance(total_return_dollars, (int, float)) and total_return_dollars < 0
        else "neu"
    )

    select_href = f"?select_ticker={html.escape(ticker)}"
    card_html = (
        "<div class='pulse-card-wrap'>"
        f"<a class='pulse-card-open' href='{select_href}' target='_self' aria-label='Open {html.escape(ticker)} in Brain'></a>"
        "<div class='pulse-card'>"
        "<div class='pulse-top'>"
        "<div class='pulse-left'>"
        f"<div class='pulse-ticker'>{html.escape(ticker)}</div>"
        f"<div class='pulse-name'>{html.escape(company)}</div>"
        "</div>"
        "<div class='pulse-mid'>"
        f"<div class='pulse-price'>{html.escape(price_text)}</div>"
        "<div class='pulse-quote-lines'>"
        f"{today_line_html}"
        f"{ah_line_html}"
        "</div>"
        "</div>"
        "<div class='pulse-right'>"
        f"<div class='pulse-pill {html.escape(pill_class)}'>{html.escape(action_ui)} {html.escape(pill_arrow)}</div>"
        f"<div class='pulse-sparkline'>{sparkline_svg}</div>"
        "</div>"
        "</div>"
        "<div class='pulse-stats'>"
        "<div class='pulse-metrics-grid'>"
        "<div class='pulse-metric-item'>"
        "<div class='pulse-metric-label muted'>Shares</div>"
        f"<div class='pulse-metric-value'>{quantity:,.2f}</div>"
        "</div>"
        "<div class='pulse-metric-item'>"
        "<div class='pulse-metric-label muted'>Market Value</div>"
        f"<div class='pulse-metric-value'>{html.escape(market_text)}</div>"
        "</div>"
        "<div class='pulse-metric-item'>"
        "<div class='pulse-metric-label muted'>Average Cost</div>"
        f"<div class='pulse-metric-value'>{html.escape(format_money(avg_cost))}</div>"
        "</div>"
        "<div class='pulse-metric-item'>"
        "<div class='pulse-metric-label muted'>Today\u2019s Return</div>"
        f"<div class='pulse-metric-value {today_return_class}'>{html.escape(today_return_pct_text)}</div>"
        "</div>"
        "<div class='pulse-metric-item'>"
        "<div class='pulse-metric-label muted'>Total Return</div>"
        f"<div class='pulse-metric-value {total_return_class}'>{html.escape(total_return_text)}</div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
    action_col_left, action_col_right = st.columns([4.8, 1.2], vertical_alignment="center")
    with action_col_right:
        if st.button("🗑", key=f"pulse_delete_btn_{ticker}", help="Delete holding", width="content"):
            _remove_holding_and_refresh(ticker)


def _remove_holding_and_refresh(ticker: str) -> None:
    updated = remove_holding(ticker)
    selected = str(st.session_state.get("selected_ticker", "")).upper()
    if selected == ticker:
        remaining = list(getattr(updated, "holdings", []) or [])
        st.session_state["selected_ticker"] = remaining[0].ticker if remaining else ""
    st.rerun()


def _get_chart_fetcher() -> ChartFetcher:
    return ChartFetcher(cache_dir=".cache/charts", stale_first=True)


def _chart_series_to_dict(series: Any) -> dict[str, Any]:
    if series is None:
        return {"bars": [], "source": "none", "quality_flags": {"MISSING"}, "error": "chart_fetch_failed"}
    bars_obj = getattr(series, "bars", None)
    if not isinstance(bars_obj, list):
        return {"bars": [], "source": "none", "quality_flags": {"MISSING"}, "error": "chart_fetch_failed"}
    return {
        "bars": [
            {
                "ts": bar.ts.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": None if bar.volume is None else float(bar.volume),
            }
            for bar in bars_obj
        ],
        "source": str(getattr(series, "source", "none")),
        "quality_flags": {str(flag) for flag in (getattr(series, "quality_flags", set()) or set())},
        "error": getattr(series, "error", None),
    }


def _downsample_bars(bars: list[dict], max_points: int = 60) -> list[dict]:
    if len(bars) <= max_points:
        return list(bars)
    if max_points <= 1:
        return [bars[-1]]
    stride = max(1, len(bars) // max_points)
    sampled = bars[::stride]
    if sampled[-1] is not bars[-1]:
        sampled = sampled + [bars[-1]]
    if len(sampled) > max_points:
        sampled = sampled[: max_points - 1] + [bars[-1]]
    return sampled


def _validate_manage_inputs(ticker_raw: str, avg_raw: str, qty_raw: str) -> str | None:
    ticker = _normalize_ticker(ticker_raw)
    if not ticker:
        return "Enter a valid ticker (A-Z, 0-9, '.' or '-')."

    if str(avg_raw).strip() == "" or str(qty_raw).strip() == "":
        return "Avg Cost and Quantity are required."

    try:
        avg = float(avg_raw)
        qty = float(qty_raw)
    except ValueError:
        return "Avg Cost and Quantity must be numeric values."

    if avg < 0 or qty < 0:
        return "Avg Cost and Quantity must be non-negative."
    return None


def _validate_ticker_exists(ticker: str) -> tuple[bool, str]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return False, "Enter a ticker symbol."

    # Primary check: lightweight live Yahoo chart endpoint (no yfinance dependency).
    try:
        provider = YahooChartMarketDataProvider()
        payload = provider.get_ohlcv(ticker=symbol, interval="1d", lookback_days=5)
        bars = payload.get("bars", []) if isinstance(payload, dict) else []
        if isinstance(bars, list) and len(bars) > 0:
            return True, ""
    except Exception:
        pass

    # Secondary checks: existing quote/chart probes.
    fetcher = _get_chart_fetcher()
    try:
        quote = get_quote_snapshot(ticker=symbol, fetcher=fetcher)
        q = quote_snapshot_to_dict(quote)
        for key in ("latest_price", "close_price", "last_regular", "after_hours_price"):
            if isinstance(q.get(key), (int, float)):
                return True, ""
    except Exception:
        pass

    try:
        series = fetcher.fetch_chart_series(ticker=symbol, range_key="1W")
        if list(series.bars):
            return True, ""
    except Exception:
        pass

    # Fallback for offline/degraded environments: local universe snapshot.
    if symbol in _load_local_universe():
        return True, ""

    return False, f"Ticker '{symbol}' not found. Check symbol and try again (Apple is AAPL)."


def _load_local_universe() -> set[str]:
    path = Path("app/data/sp500_universe.json")
    if not path.exists():
        return set(_LOCAL_UNIVERSE_FALLBACK)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(_LOCAL_UNIVERSE_FALLBACK)

    tickers: set[str] = set()
    if isinstance(data, list):
        tickers = {str(x).strip().upper() for x in data if str(x).strip()}
    elif isinstance(data, dict):
        raw = data.get("tickers", [])
        if isinstance(raw, list):
            tickers = {str(x).strip().upper() for x in raw if str(x).strip()}
    if not tickers:
        return set(_LOCAL_UNIVERSE_FALLBACK)
    return tickers


def _normalize_ticker(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if not _TICKER_PATTERN.fullmatch(text):
        return None
    return text


def _holding_value(holding: Any, key: str) -> Any:
    if isinstance(holding, dict):
        return holding.get(key)
    return getattr(holding, key, None)


def _holding_to_dict(holding: Any) -> dict[str, Any]:
    if isinstance(holding, dict):
        return holding
    if hasattr(holding, "model_dump"):
        try:
            return holding.model_dump(exclude_none=True)
        except Exception:
            pass
    return {
        "ticker": _holding_value(holding, "ticker"),
        "avg_cost": _holding_value(holding, "avg_cost"),
        "quantity": _holding_value(holding, "quantity"),
    }


def _filter_intraday_today(bars: list[dict], tz_name: str = "America/New_York") -> list[dict]:
    if not bars:
        return []

    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    by_ts: dict[datetime, dict] = {}

    for bar in bars:
        if not isinstance(bar, dict):
            continue
        ts_val = bar.get("ts")
        dt = _parse_ts(ts_val)
        if dt is None:
            continue
        local_dt = dt.astimezone(tz)
        if local_dt.date() != today:
            continue

        try:
            close = float(bar.get("close"))
        except (TypeError, ValueError):
            continue

        normalized = {
            "ts": local_dt.isoformat(),
            "open": _to_float(bar.get("open"), close),
            "high": _to_float(bar.get("high"), close),
            "low": _to_float(bar.get("low"), close),
            "close": close,
            "volume": _to_optional_float(bar.get("volume")),
        }
        by_ts[local_dt] = normalized

    ordered = [by_ts[key] for key in sorted(by_ts.keys())]
    return ordered


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _last_close_or_none(bars: Any) -> float | None:
    if not isinstance(bars, list) or not bars:
        return None
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    try:
        return float(last.get("close"))
    except (TypeError, ValueError):
        return None


def _money_or_dash(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "--"
    return format_money(value)


def _sparkline_points_from_bars(bars: list[dict], n: int = 30) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    subset = bars[-n:] if len(bars) > n else bars
    for bar in subset:
        if not isinstance(bar, dict):
            continue
        ts = str(bar.get("ts", "")).strip()
        try:
            close = float(bar.get("close"))
        except (TypeError, ValueError):
            continue
        points.append((ts, close))
    return points


def _build_sparkline_svg(sparkline_points: list[tuple[str, float]], action: str) -> str:
    width = 150
    height = 52
    color = "#21e06d" if action == "ACCUMULATE" else "#ef5350" if action == "REDUCE" else "#8fa3b8"

    if len(sparkline_points) < 2:
        return (
            f"<svg class='pulse-sparkline' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
            "<line x1='4' y1='26' x2='146' y2='26' stroke='rgba(143,163,184,0.45)' stroke-width='1.6'/>"
            "</svg>"
        )

    values = [point[1] for point in sparkline_points]
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v
    if span == 0:
        span = 1.0

    x_step = (width - 2) / max(1, len(values) - 1)
    coords: list[str] = []
    for idx, value in enumerate(values):
        x = 1 + idx * x_step
        y = (height - 2) - ((value - min_v) / span) * (height - 4)
        coords.append(f"{x:.2f},{y:.2f}")

    return (
        f"<svg class='pulse-sparkline' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        "<defs><linearGradient id='sparkFade' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='rgba(255,255,255,0.18)'/>"
        "<stop offset='100%' stop-color='rgba(255,255,255,0.02)'/>"
        "</linearGradient></defs>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(coords)}'/>"
        "</svg>"
    )


def _today_change_metrics(today_bars: list[dict], fallback_bars: list[dict]) -> tuple[float, float]:
    if len(today_bars) >= 2:
        try:
            start_close = float(today_bars[0].get("close", 0.0))
            end_close = float(today_bars[-1].get("close", 0.0))
        except (TypeError, ValueError):
            start_close = 0.0
            end_close = 0.0
    elif len(fallback_bars) >= 2:
        try:
            start_close = float(fallback_bars[-2].get("close", 0.0))
            end_close = float(fallback_bars[-1].get("close", 0.0))
        except (TypeError, ValueError):
            start_close = 0.0
            end_close = 0.0
    else:
        return 0.0, 0.0

    if start_close == 0.0:
        return 0.0, 0.0
    delta = end_close - start_close
    pct = (delta / start_close) * 100.0
    return delta, pct


def _build_badge_row_html(badge_state: dict) -> str:
    age = int(max(0, int(badge_state.get("age_minutes", 0) or 0)))
    spans = [
        f"<span class='badge-age' title='{html.escape(TOOLTIP_AGE)}'>Age: {age}m</span>",
    ]
    if bool(badge_state.get("show_degraded", False)):
        spans.append(f"<span class='badge-icon' title='{html.escape(TOOLTIP_DEGRADED)}'>⚠</span>")
    if bool(badge_state.get("show_tool_down", False)):
        spans.append(f"<span class='badge-icon' title='{html.escape(TOOLTIP_TOOL_DOWN)}'>⛔</span>")
    return f"<div class='badge-row'>{''.join(spans)}</div>"


def _format_quote_ts(value: Any) -> str:
    dt = _parse_ts(value)
    if dt is None:
        return ""
    local = dt.astimezone(ZoneInfo("America/New_York"))
    return local.strftime("%H:%M ET")


def _pulse_pill_class(action_raw: str) -> str:
    mapping = {
        "ACCUMULATE": "pulse-pill-buy",
        "WAIT": "pulse-pill-hold",
        "REDUCE": "pulse-pill-sell",
    }
    return mapping.get(action_raw, "pulse-pill-hold")


def _pill_arrow(action_ui: str) -> str:
    return {"BUY": "↗", "HOLD": "→", "SELL": "↘"}.get(action_ui, "→")


def _company_name(ticker: str) -> str:
    known = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "GOOG": "Alphabet Inc.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms",
        "TSLA": "Tesla Inc.",
        "PLUG": "Plug Power Inc.",
    }
    return known.get(ticker, "")


def _fallback_context_pack(ticker: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "meta": {
            "ticker": ticker,
            "generated_at": now,
            "data_quality": {
                "prices": {
                    "as_of": now,
                    "now": now,
                    "age_minutes": 0.0,
                    "stale": True,
                    "stale_minutes_threshold": 90,
                },
                "indicators": {
                    "as_of": now,
                    "now": now,
                    "age_minutes": 0.0,
                    "stale": True,
                    "stale_minutes_threshold": 90,
                },
                "overall_stale": True,
                "notes": ["TOOL_DOWN", "PRICE_FETCH_FAILED"],
            },
        },
        "prices": {"as_of": now, "bars": []},
        "indicators": {"as_of": now, "metrics": {}},
        "drl": {
            "result": {
                "action_final": "WAIT",
                "confidence_cap": 0,
                "regime_1D": "NEUTRAL",
                "regime_1W": "NEUTRAL",
                "gates_triggered": [],
                "conflicts": ["TOOL_DOWN"],
                "decision_trace": {
                    "ticker": ticker,
                    "action_final": "WAIT",
                    "base_action": "WAIT",
                    "score_final": 0.0,
                },
            }
        },
    }
