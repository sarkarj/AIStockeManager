from __future__ import annotations

import html
import re
import warnings
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.utils.formatting import format_int, format_money
from app.ui.viewmodels.brain_market_vm import BrainMarketVM, Gauge

_HEDGE_TERMS = {"may", "might", "could", "possibly", "suggests", "likely"}

GAUGE_TOOLTIPS = {
    "TREND": "EMA50 vs SMA200 vs price structure, plus trend-direction context.",
    "MOMENTUM": "RSI + MACD (+ signal) + Stoch indicate momentum alignment.",
    "RISK": "ATR% and overbought/oversold risk context (presentation-only).",
    "STRENGTH": "ADX (trend strength) + VROC (volume participation) with contextual strength view.",
}


def render_brain_market_card(vm: BrainMarketVM, range_key: str) -> None:
    with st.container(border=True):
        _render_price_chart(vm=vm, range_key=range_key)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _render_quote_stats(vm)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _render_gauges(vm.gauges)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _render_why_block(vm.why_block)


def _render_price_chart(vm: BrainMarketVM, range_key: str) -> None:
    frame = _bars_to_frame(vm)
    label = "1D (RTH+Extended)" if str(range_key).upper() == "1D" else str(range_key).upper()
    st.markdown(f"<div class='tiny muted'>{html.escape(label)}</div>", unsafe_allow_html=True)

    if frame.empty:
        st.info("No chart data available (live and cache empty).")
        return

    figure = _build_price_figure(frame=frame, title=f"{vm.ticker} {range_key}", advanced=(str(range_key).upper() == "ADVANCED"))
    if figure is None:
        st.info("Plot rendering unavailable.")
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*keyword arguments have been deprecated.*Use `config` instead.*",
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"brain_market_chart_{vm.ticker}_{range_key}",
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "responsive": True,
            },
        )


def _render_quote_stats(vm: BrainMarketVM) -> None:
    quote = vm.quote
    currency = str(getattr(quote, "currency", "USD") or "USD")
    volume_text = format_int(quote.volume)
    rows = [
        ("Open", _money_or_dash(quote.open, currency=currency), "Volume", volume_text),
        ("Day Low", _money_or_dash(quote.day_low, currency=currency), "Day High", _money_or_dash(quote.day_high, currency=currency)),
        ("Year Low", _money_or_dash(quote.year_low, currency=currency), "Year High", _money_or_dash(quote.year_high, currency=currency)),
    ]
    row_html = "".join(
        (
            "<tr>"
            f"<td>{html.escape(str(m1))}</td>"
            f"<td>{html.escape(str(v1))}</td>"
            f"<td>{html.escape(str(m2))}</td>"
            f"<td>{html.escape(str(v2))}</td>"
            "</tr>"
        )
        for m1, v1, m2, v2 in rows
    )
    table_html = (
        "<table class='brain-quote-compact'>"
        "<tbody>"
        f"{row_html}"
        "</tbody>"
        "</table>"
    )
    with st.expander("Quote / Stats", expanded=False):
        st.markdown(table_html, unsafe_allow_html=True)
        if quote.notes:
            st.caption(f"stats_notes: {', '.join(quote.notes)}")


def _render_gauges(gauges: list[Gauge]) -> None:
    blocks = []
    for gauge in gauges[:4]:
        clamped = max(0.0, min(100.0, float(gauge.score)))
        angle = -90.0 + (clamped * 1.8)
        tone_class = (
            "gauge-good"
            if str(gauge.tone) == "good"
            else "gauge-bad"
            if str(gauge.tone) == "bad"
            else "gauge-neutral"
        )
        tooltip = html.escape(GAUGE_TOOLTIPS.get(str(gauge.label), ""))
        blocks.append(
            f"<div class='brain-gauge {tone_class}' title='{tooltip}'>"
            f"<div class='brain-gauge-label' title='{tooltip}'>{html.escape(gauge.label)}</div>"
            "<div class='brain-gauge-dial'>"
            f"<div class='brain-gauge-needle' style='transform: rotate({angle:.1f}deg);'></div>"
            "</div>"
            f"<div class='brain-gauge-value'>{html.escape(gauge.value)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='brain-gauges'>{''.join(blocks)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny muted'>Gauges based on 1D indicators.</div>", unsafe_allow_html=True)


def _render_why_block(why_block: dict[str, Any]) -> None:
    tip = (
        "1) DRL computes action + capped confidence from indicators and gates/conflicts. "
        "2) LLM summarizes drivers/risks from provided inputs. "
        "3) If hub is invalid/degraded, deterministic fallback is used."
    )
    narrative_label = "Narrative (LLM)" if bool(why_block.get("hub_valid")) else "Narrative (fallback)"
    st.markdown(
        "<div class='why-title'>"
        f"<span title='{html.escape(tip)}'>🧠 The Why</span> "
        f"<span class='why-ai-label'>{html.escape(narrative_label)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    mode = str(why_block.get("mode", "")).upper()
    metrics = why_block.get("metrics", {}) if isinstance(why_block.get("metrics"), dict) else {}
    if mode == "HUB":
        one_liner = _clean_text(str(why_block.get("one_liner", "")), allow_hedge=False)
        action_raw = str(why_block.get("drl_action_raw", "WAIT")).upper()
        if action_raw == "WAIT" and _contains_strong_directional(one_liner):
            one_liner = "Current setup supports a WAIT posture while conditions remain mixed."
        st.markdown(f"- 🤝 {one_liner}")
        _render_emoji_bullets(
            title="",
            items=why_block.get("drivers", []),
            emojis=["📈", "🧭", "⚡", "🧱"],
            limit=3,
            allow_hedge=False,
        )
        risk_items = why_block.get("conflicts") or why_block.get("watch") or []
        for line in _build_plain_risk_lines(metrics=metrics, fallback_items=risk_items):
            st.markdown(f"- ⚠️ {line}")
        return

    title = str(why_block.get("title", "The Why (Deterministic fallback)"))
    reason = _clean_text(str(why_block.get("reason", "Context pack missing hub artifact")), allow_hedge=True)
    action_final = str(why_block.get("action_final", "WAIT"))
    confidence_cap = float(why_block.get("confidence_cap", 0.0) or 0.0)
    st.markdown(f"- ⚠️ {title}: {reason}")
    st.markdown(f"- 🤝 DRL action/confidence: {action_final} / {confidence_cap:.0f}")
    for line in why_block.get("drivers", [])[:3]:
        st.markdown(f"- 📈 {_clean_text(str(line), allow_hedge=False)}")
    for line in _build_plain_risk_lines(metrics=metrics, fallback_items=why_block.get("conflicts", [])):
        st.markdown(f"- ⚠️ {line}")


def _render_emoji_bullets(
    title: str,
    items: Any,
    emojis: list[str],
    limit: int,
    allow_hedge: bool,
) -> None:
    if not isinstance(items, list) or not items:
        return
    if str(title).strip():
        st.markdown(f"**{title}**")
    for idx, item in enumerate(items[:limit]):
        if not isinstance(item, dict):
            continue
        text = _clean_text(str(item.get("text", "")), allow_hedge=allow_hedge)
        emoji = emojis[min(idx, len(emojis) - 1)]
        chips = _citation_chip_suffix(item.get("citations", []))
        st.markdown(f"- {emoji} {text}{chips}", unsafe_allow_html=True)


def _normalize_refs(refs: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    if not isinstance(refs, list):
        return ordered
    for ref in refs:
        raw = str(ref).strip()
        if not raw:
            continue
        raw = _strip_ref_prefixes(raw)
        for token in [chunk.strip() for chunk in raw.split(",") if chunk.strip()]:
            clean = _strip_ref_prefixes(token)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(_pretty_ref_label(clean))
    return ordered


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


def _clean_text(value: str, allow_hedge: bool) -> str:
    text = str(value).replace("\\n", "\n")
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    compact = " ".join(parts)
    if allow_hedge:
        return compact
    return _remove_hedge_terms(compact)


def _bars_to_frame(vm: BrainMarketVM) -> pd.DataFrame:
    if not vm.chart_series.bars:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    rows = [
        {
            "ts": bar.ts,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": None if bar.volume is None else float(bar.volume),
        }
        for bar in vm.chart_series.bars
    ]
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return frame


def _build_price_figure(frame: pd.DataFrame, title: str, advanced: bool):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

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
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
        paper_bgcolor="#101826",
        plot_bgcolor="#101826",
        font={"color": "#dce7f3", "size": 12},
        xaxis={"gridcolor": "rgba(255,255,255,0.04)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.04)", "side": "right"},
        height=320,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def _money_or_dash(value: float | None, *, currency: str = "USD") -> str:
    return format_money(value, currency=currency)


def _remove_hedge_terms(text: str) -> str:
    cleaned = str(text)
    for term in sorted(_HEDGE_TERMS, key=len, reverse=True):
        cleaned = re.sub(rf"(?i)\b{re.escape(term)}\b", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _contains_strong_directional(text: str) -> bool:
    lowered = str(text).lower()
    patterns = ["strong buy", "buy now", "strong sell", "sell now", "aggressive buy", "aggressive sell"]
    return any(pat in lowered for pat in patterns)


def _pretty_ref_label(ref: str) -> str:
    normalized = str(ref).strip()
    if normalized.startswith("indicator:"):
        return normalized.split(":", 1)[1]
    if normalized.startswith("news:"):
        return f"news#{normalized.split(':', 1)[1]}"
    if normalized.startswith("macro:"):
        return f"macro#{normalized.split(':', 1)[1]}"
    return normalized


def _build_plain_risk_lines(metrics: dict[str, Any], fallback_items: Any) -> list[str]:
    lines: list[str] = []
    rsi = _to_float_or_none(metrics.get("rsi_14"))
    if rsi is not None:
        zone = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        lines.append(
            f"RSI is {rsi:.1f} ({zone}). A move below 45 weakens momentum; above 65 strengthens momentum."
        )
    macd = _to_float_or_none(metrics.get("macd"))
    macd_signal = _to_float_or_none(metrics.get("macd_signal"))
    if macd is not None and macd_signal is not None:
        if macd < macd_signal:
            lines.append(
                f"MACD ({macd:.2f}) is below its signal ({macd_signal:.2f}), so momentum is not confirmed yet; watch for a cross above the signal."
            )
        else:
            lines.append(
                f"MACD ({macd:.2f}) is above its signal ({macd_signal:.2f}), which supports momentum while that relationship holds."
            )
    atr = _to_float_or_none(metrics.get("atr_pct"))
    if atr is not None:
        lines.append(f"ATR% is {atr:.2f}; higher volatility can trigger tighter risk controls and confidence caps.")

    if isinstance(fallback_items, list):
        for item in fallback_items:
            if not isinstance(item, dict):
                continue
            text = _clean_text(str(item.get("text", "")), allow_hedge=False)
            if not text:
                continue
            if text.lower() not in {line.lower() for line in lines}:
                lines.append(text)
            if len(lines) >= 3:
                break
    return lines[:3]


def _citation_chip_suffix(citations: Any, *, max_items: int = 5) -> str:
    refs = _normalize_refs(citations if isinstance(citations, list) else [])
    if not refs:
        return ""
    shown = refs[:max_items]
    remainder = len(refs) - len(shown)
    chips = " ".join(f"<span class='chip'>{html.escape(ref)}</span>" for ref in shown)
    if remainder > 0:
        chips = (
            f"{chips} <span class='chip' title='{html.escape(', '.join(refs[max_items:]))}'>+{remainder} more</span>"
        )
    return f" {chips}"


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
