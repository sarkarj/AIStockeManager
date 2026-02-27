from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from app.core.marketdata.query_graph import MarketQueryService
from app.ui.components.ui_utils import format_money, format_pct, safe_get
from app.ui.viewmodels.brain_vm import build_brain_view_model

_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
_UNIVERSE_PATH = Path("app/data/sp100_universe.json")
_ET = ZoneInfo("America/New_York")
_RTH_OPEN = dt_time(9, 30)
_RTH_CLOSE = dt_time(16, 0)
_HORIZON_METRIC_LABELS = {
    "today_after": "Today + After-hours",
    "regular": "Regular Session",
}

# Fallback list used only when sp100_universe.json is unavailable.
_SP100_FALLBACK = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA",
    "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CHTR", "CL",
    "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "D", "DHR",
    "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG",
    "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "JNJ", "JPM", "KHC",
    "KMI", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDT", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA",
    "ORCL", "OXY", "PANW", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
    "SBUX", "SCHW", "SLB", "SO", "SPG", "T", "TGT", "TMO", "TXN", "UNH",
    "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "AVGO",
]

_COMPANY_FALLBACK = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com Inc.",
    "GOOG": "Alphabet Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
}

def render_horizon(
    context_loader: Any | None = None,
    market_query: MarketQueryService | None = None,
) -> None:
    st.markdown("<div class='card'><div class='section-title'>The Horizon</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='pulse-block-gap'></div>", unsafe_allow_html=True)
    st.session_state.setdefault("horizon_metric_mode", "today_after")
    metric_mode = str(st.session_state.get("horizon_metric_mode", "today_after")).strip().lower()
    if metric_mode not in _HORIZON_METRIC_LABELS:
        metric_mode = "today_after"
        st.session_state["horizon_metric_mode"] = metric_mode
    st.selectbox(
        "Mover metric",
        options=["today_after", "regular"],
        format_func=lambda value: _HORIZON_METRIC_LABELS.get(str(value), str(value)),
        key="horizon_metric_mode",
        label_visibility="collapsed",
    )

    query = market_query or st.session_state.get("_market_query")
    if not isinstance(query, MarketQueryService):
        query = MarketQueryService(cache_dir=".cache/charts", context_loader=context_loader)

    universe = _load_sp100_universe()
    movers = _load_horizon_movers(
        universe=universe,
        market_query=query,
        limit=10,
        metric_mode=metric_mode,
    )
    if not movers:
        st.info("No mover data available right now.")
        return

    for row in movers:
        ticker = str(row["ticker"]).upper()
        action_ui = str(row.get("ui_action_label", "HOLD"))
        action_raw = str(row.get("drl_action_raw", "WAIT"))
        pill_class = _pill_class(action_raw)
        confidence = int(round(float(row.get("confidence_cap", 0.0) or 0.0)))
        age_m = int(max(0, int(row.get("age_minutes", 0) or 0)))
        sparkline_svg = _build_sparkline_svg(
            values=[float(v) for v in row.get("sparkline_values", []) if isinstance(v, (int, float))],
            positive=bool(row.get("primary_abs", 0.0) >= 0),
        )
        quote_payload = row.get("quote", {}) if isinstance(row.get("quote"), dict) else {}
        show_extended = bool(quote_payload.get("show_extended_session", False))
        extended_label = str(quote_payload.get("extended_label", "") or "").strip() or "After-hours"
        extended_icon = "🌙" if extended_label == "After-hours" else "🌅"
        metric_label = str(row.get("metric_label", "Today + After-hours"))
        if not show_extended and "After-hours" in metric_label:
            metric_label = "Today"
        today_line = _build_change_line(
            abs_value=row.get("primary_abs"),
            pct_value=row.get("primary_pct"),
            label=metric_label,
        )
        ah_abs = quote_payload.get("after_hours_change_abs", row.get("after_hours_abs"))
        ah_pct = quote_payload.get("after_hours_change_pct", row.get("after_hours_pct"))
        ah_line = _build_change_line(
            abs_value=ah_abs,
            pct_value=ah_pct,
            label=f"{extended_icon} {extended_label}",
        )
        today_line_html = today_line if today_line is not None else "<div class='quote-line neu'>Today: —</div>"
        if show_extended and ah_line is None:
            ah_line = f"<div class='quote-line neu'>{html.escape(f'{extended_icon} {extended_label}: —')}</div>"
        if not show_extended:
            ah_line = ""

        card_html = (
            "<div class='horizon-card-wrap'>"
            f"<a class='horizon-card-open' href='?select_ticker={html.escape(ticker)}' target='_self' "
            f"aria-label='Open {html.escape(ticker)} in Brain'></a>"
            "<div class='pulse-card horizon-pulse-card'>"
            "<div class='pulse-top'>"
            "<div class='pulse-left'>"
            f"<div class='pulse-ticker'>{html.escape(ticker)}</div>"
            f"<div class='pulse-name'>{html.escape(_company_name(ticker))}</div>"
            "</div>"
            "<div class='pulse-mid'>"
            f"<div class='pulse-price'>{html.escape(format_money(row.get('latest_price')))}</div>"
            "<div class='pulse-quote-lines'>"
            f"{today_line_html}"
            f"{ah_line}"
            "</div>"
            "</div>"
            "<div class='pulse-right'>"
            f"<div class='pulse-pill {html.escape(pill_class)}'>{html.escape(action_ui)} {_pill_arrow(action_ui)}</div>"
            f"<div class='pulse-sparkline'>{sparkline_svg}</div>"
            f"<div class='pulse-meta-line'>Conf {confidence} · Age {age_m}m</div>"
            "</div>"
            "</div>"
            "</div>"
            "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)


def _load_horizon_movers(
    universe: list[str],
    market_query: MarketQueryService,
    limit: int,
    metric_mode: str,
) -> list[dict[str, Any]]:
    cache_key = f"horizon_movers_cache_v2:{str(metric_mode).strip().lower()}"
    cached = st.session_state.get(cache_key)
    now_ts = time.time()
    if isinstance(cached, dict):
        rows = cached.get("rows")
        fetched_at = cached.get("fetched_at")
        if isinstance(rows, list) and isinstance(fetched_at, (int, float)) and (now_ts - float(fetched_at) <= 900.0):
            return rows

    fresh = _build_top_movers(
        universe=universe,
        market_query=market_query,
        limit=limit,
        metric_mode=metric_mode,
    )
    if fresh:
        st.session_state[cache_key] = {"rows": fresh, "fetched_at": now_ts}
        return fresh

    if isinstance(cached, dict) and isinstance(cached.get("rows"), list):
        return cached.get("rows", [])
    return []


def _build_top_movers(
    universe: list[str],
    market_query: MarketQueryService,
    limit: int,
    metric_mode: str,
) -> list[dict[str, Any]]:
    mode = str(metric_mode or "today_after").strip().lower()
    if mode not in _HORIZON_METRIC_LABELS:
        mode = "today_after"
    rows: list[dict[str, Any]] = []
    for ticker in universe:
        symbol = _normalize_ticker(ticker)
        if not symbol:
            continue
        try:
            series = market_query.chart_series(ticker=symbol, range_key="1W")
        except Exception:
            continue

        snapshot = _snapshot_from_weekly_series(series)
        latest = _to_float(snapshot.get("latest_price"))
        prev_close = _to_float(snapshot.get("prev_close_price"))
        last_regular = _to_float(snapshot.get("last_regular"))
        if latest is None or prev_close is None or prev_close == 0:
            continue

        regular_abs = None
        regular_pct = None
        if last_regular is not None:
            regular_abs = last_regular - prev_close
            regular_pct = (regular_abs / prev_close) * 100.0

        today_after_abs = latest - prev_close
        today_after_pct = (today_after_abs / prev_close) * 100.0
        after_hours_abs = None
        after_hours_pct = None
        if isinstance(last_regular, (int, float)) and last_regular != 0 and isinstance(latest, (int, float)):
            if abs(latest - last_regular) > 0:
                after_hours_abs = latest - last_regular
                after_hours_pct = (after_hours_abs / last_regular) * 100.0

        if mode == "regular":
            primary_abs = regular_abs
            primary_pct = regular_pct
            metric_label = "Today (Regular)"
        else:
            primary_abs = today_after_abs
            primary_pct = today_after_pct
            metric_label = "Today + After-hours"

        if not isinstance(primary_abs, (int, float)) or not isinstance(primary_pct, (int, float)):
            continue

        rows.append(
            {
                "ticker": symbol,
                "latest_price": latest,
                "primary_abs": primary_abs,
                "primary_pct": primary_pct,
                "metric_label": metric_label,
                "today_abs": today_after_abs,
                "today_pct": today_after_pct,
                "regular_abs": regular_abs,
                "regular_pct": regular_pct,
                "after_hours_abs": after_hours_abs,
                "after_hours_pct": after_hours_pct,
                "age_minutes": int(snapshot.get("age_minutes", 0) or 0),
                "sparkline_values": [],
                "ui_action_label": "HOLD",
                "drl_action_raw": "WAIT",
                "confidence_cap": 0.0,
                "quote": {
                    "latest_price": latest,
                    "prev_close_price": prev_close,
                    "close_price": last_regular,
                    "today_change_abs": today_after_abs,
                    "today_change_pct": today_after_pct,
                    "after_hours_change_abs": after_hours_abs,
                    "after_hours_change_pct": after_hours_pct,
                    "source": str(snapshot.get("source", "none")),
                    "quality_flags": list(snapshot.get("quality_flags", [])),
                },
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -abs(float(row["primary_pct"])),
            -float(row["primary_pct"]),
            str(row["ticker"]),
        ),
    )
    top = ranked[: max(1, int(limit))]
    for row in top:
        symbol = str(row.get("ticker", "")).upper()
        sparkline_values: list[float] = []
        try:
            series_1d = market_query.chart_series(ticker=symbol, range_key="1D")
            sparkline_values = [float(bar.close) for bar in series_1d.bars]
        except Exception:
            sparkline_values = []
        row["sparkline_values"] = _downsample_values(sparkline_values, max_points=40)
        try:
            quote = market_query.quote_snapshot_dict(ticker=symbol)
        except Exception:
            quote = {}
        if isinstance(quote, dict):
            row["quote"] = quote
            row["age_minutes"] = _quote_age_minutes(quote)
            latest_quote = _to_float(quote.get("latest_price"))
            if isinstance(latest_quote, float):
                row["latest_price"] = latest_quote
            ah_abs_quote = _to_float(quote.get("after_hours_change_abs"))
            ah_pct_quote = _to_float(quote.get("after_hours_change_pct"))
            if ah_abs_quote is not None:
                row["after_hours_abs"] = ah_abs_quote
            if ah_pct_quote is not None:
                row["after_hours_pct"] = ah_pct_quote

        try:
            context_pack = market_query.short_context(ticker=symbol)
            vm = build_brain_view_model(context_pack=context_pack, quote=row.get("quote"))
            row["ui_action_label"] = str(vm.get("ui_action_label", "HOLD"))
            row["drl_action_raw"] = str(vm.get("drl_action_raw", "WAIT"))
            row["confidence_cap"] = float(vm.get("confidence_cap", 0.0) or 0.0)
        except Exception:
            row["ui_action_label"] = "HOLD"
            row["drl_action_raw"] = "WAIT"
            row["confidence_cap"] = 0.0
    return top


def _load_sp100_universe() -> list[str]:
    fallback = list(_SP100_FALLBACK)
    if _UNIVERSE_PATH.exists():
        try:
            data = json.loads(_UNIVERSE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                vals = [_normalize_ticker(x) for x in data]
                cleaned = [v for v in vals if v]
                if len(cleaned) >= 100:
                    return cleaned[:100]
            if isinstance(data, dict):
                raw = data.get("tickers", [])
                if isinstance(raw, list):
                    vals = [_normalize_ticker(x) for x in raw]
                    cleaned = [v for v in vals if v]
                    if len(cleaned) >= 100:
                        return cleaned[:100]
        except Exception:
            pass
    return fallback[:100]


def _normalize_ticker(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if not _TICKER_PATTERN.fullmatch(text):
        return None
    return text


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quote_age_minutes(quote: dict[str, Any]) -> int:
    now = datetime.now(timezone.utc)
    ts_candidates = [quote.get("latest_ts"), quote.get("close_ts"), quote.get("as_of_ts")]
    for candidate in ts_candidates:
        if not candidate:
            continue
        text = str(candidate)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = int((now - parsed.astimezone(timezone.utc)).total_seconds() // 60)
        return max(0, age)
    return 0


def _snapshot_from_weekly_series(series: Any) -> dict[str, Any]:
    bars = list(getattr(series, "bars", []) or [])
    if not bars:
        return {
            "latest_price": None,
            "prev_close_price": None,
            "last_regular": None,
            "age_minutes": 0,
            "source": str(getattr(series, "source", "none")),
            "quality_flags": [str(x) for x in (getattr(series, "quality_flags", set()) or set())],
        }

    bars = sorted(bars, key=lambda bar: bar.ts)
    latest_bar = bars[-1]
    latest_price = float(latest_bar.close)

    session_closes: dict[datetime.date, Any] = {}
    for bar in bars:
        local = bar.ts.astimezone(_ET)
        if _RTH_OPEN <= local.time() <= _RTH_CLOSE:
            session_closes[local.date()] = bar
    ordered_days = sorted(session_closes.keys())
    last_regular = float(session_closes[ordered_days[-1]].close) if ordered_days else None
    prev_close = float(session_closes[ordered_days[-2]].close) if len(ordered_days) >= 2 else None

    now_utc = datetime.now(timezone.utc)
    age_minutes = int((now_utc - latest_bar.ts.astimezone(timezone.utc)).total_seconds() // 60)
    return {
        "latest_price": latest_price,
        "prev_close_price": prev_close,
        "last_regular": last_regular,
        "age_minutes": max(0, age_minutes),
        "source": str(getattr(series, "source", "none")),
        "quality_flags": [str(x) for x in (getattr(series, "quality_flags", set()) or set())],
    }


def _build_change_line(abs_value: Any, pct_value: Any, label: str) -> str | None:
    abs_num = _to_float(abs_value)
    pct_num = _to_float(pct_value)
    if abs_num is None or pct_num is None:
        return None
    arrow = "▲" if abs_num >= 0 else "▼"
    sign = "+" if abs_num >= 0 else ""
    tone = "pos" if abs_num > 0 else "neg" if abs_num < 0 else "neu"
    text = f"{arrow} {sign}{format_money(abs_num)} ({format_pct(pct_num)}) {label}"
    return f"<div class='quote-line {tone}'>{html.escape(text)}</div>"


def _downsample_values(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return list(values)
    if max_points <= 1:
        return [values[-1]]
    stride = max(1, len(values) // max_points)
    sampled = values[::stride]
    if sampled[-1] != values[-1]:
        sampled = sampled + [values[-1]]
    if len(sampled) > max_points:
        sampled = sampled[: max_points - 1] + [values[-1]]
    return sampled


def _build_sparkline_svg(values: list[float], positive: bool) -> str:
    if len(values) < 2:
        return "<svg viewBox='0 0 140 52' class='pulse-sparkline' role='img' aria-label='No sparkline data'></svg>"
    width = 140.0
    height = 52.0
    pad = 4.0
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi != lo else 1.0
    step = (width - 2 * pad) / max(1, len(values) - 1)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = pad + idx * step
        norm = (value - lo) / span
        y = (height - pad) - norm * (height - 2 * pad)
        points.append(f"{x:.2f},{y:.2f}")
    color = "#21e06d" if positive else "#ff5b5b"
    return (
        "<svg viewBox='0 0 140 52' class='pulse-sparkline' role='img' aria-label='Horizon sparkline'>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' "
        f"points='{' '.join(points)}'></polyline>"
        "</svg>"
    )


def _pill_class(action_raw: str) -> str:
    value = str(action_raw).upper()
    if value == "ACCUMULATE":
        return "pulse-pill-buy"
    if value == "REDUCE":
        return "pulse-pill-sell"
    return "pulse-pill-hold"


def _pill_arrow(action_ui: str) -> str:
    label = str(action_ui).upper()
    if label == "BUY":
        return "↗"
    if label == "SELL":
        return "↘"
    return "→"


def _company_name(ticker: str) -> str:
    return _COMPANY_FALLBACK.get(ticker.upper(), "")
