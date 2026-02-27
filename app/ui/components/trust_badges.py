from __future__ import annotations

from typing import Any


def compute_trust_state(context_pack: dict) -> dict:
    data_quality = context_pack.get("meta", {}).get("data_quality", {})
    notes = [str(note) for note in data_quality.get("notes", [])]
    notes_blob = " ".join(notes).lower()

    stale = bool(data_quality.get("overall_stale", False))

    tool_down = False
    if "tool_down" in notes_blob:
        tool_down = True

    news_data = _extract_channel(context_pack, "news")
    macro_data = _extract_channel(context_pack, "macro")
    if isinstance(news_data, dict) and "error" in news_data:
        tool_down = True
    if isinstance(macro_data, dict) and "error" in macro_data:
        tool_down = True

    has_news = bool(isinstance(news_data, dict) and news_data.get("items"))
    has_macro = bool(isinstance(macro_data, dict) and macro_data.get("items"))

    hub_card = context_pack.get("hub_card")
    citations_valid = _hub_card_citations_valid(hub_card)

    degraded = (
        stale
        or tool_down
        or "bedrock_unavailable" in notes_blob
        or "fallback" in notes_blob
        or (isinstance(hub_card, dict) and hub_card.get("meta", {}).get("mode") == "DEGRADED")
    )

    hub_mode = ""
    if isinstance(hub_card, dict):
        hub_mode = str(hub_card.get("meta", {}).get("mode", ""))
    grounded = bool(isinstance(hub_card, dict) and citations_valid and hub_mode in {"FULL", "TECHNICAL_ONLY"})

    return {
        "grounded": grounded,
        "degraded": degraded,
        "tool_down": tool_down,
        "stale": stale,
    }


def compute_brain_trust_state(
    context_pack: dict,
    chart_series: dict | None,
    market_data_provider_up: bool,
    quote: dict | None = None,
    price_sanity_flags: list[str] | None = None,
) -> dict:
    chart = chart_series or {}
    chart_flags = [str(x).upper() for x in chart.get("flags", [])]
    chart_error = str(chart.get("diagnostics", {}).get("error", "") or "").strip()
    chart_source = str(chart.get("source", "")).strip().lower()
    quote_data = quote or {}
    quote_source = str(quote_data.get("source", "")).strip().lower()
    has_quote_payload = bool(quote_data)
    quote_flags = {str(x).upper() for x in (quote_data.get("quality_flags", []) or [])}
    sanity_flags = {str(x).upper() for x in (price_sanity_flags or [])}

    hub_status = str(context_pack.get("meta", {}).get("hub", {}).get("status", "")).strip().lower()
    hub_mode = str(context_pack.get("meta", {}).get("hub", {}).get("mode", "")).strip().upper()
    hub_valid_meta = bool(context_pack.get("meta", {}).get("hub", {}).get("hub_valid", False))
    hub_reason = str(context_pack.get("meta", {}).get("hub", {}).get("reason", "") or "").strip()
    hub_card = context_pack.get("hub_card")
    hub_valid = bool(hub_valid_meta or (isinstance(hub_card, dict) and _hub_card_citations_valid(hub_card)))

    provider_failure = _is_provider_failure_error(chart_error)
    provider_down = not bool(market_data_provider_up)
    tool_down = provider_down or provider_failure
    tool_down_reason = ""
    if provider_down:
        tool_down_reason = "Market data provider DOWN."
    elif provider_failure:
        tool_down_reason = "Price provider fetch failed."

    stale = bool(context_pack.get("meta", {}).get("data_quality", {}).get("overall_stale", False))
    chart_data_issue = any(
        flag in {"STALE", "STALE_CACHE", "MISSING", "EMPTY_LIVE", "EMPTY_CACHE", "STALE_MARKET_TS"}
        for flag in chart_flags
    )
    quote_close = quote_data.get("close_price")
    quote_after = quote_data.get("after_hours_price")
    quote_missing = has_quote_payload and (quote_source in {"", "none"} or (quote_close is None and quote_after is None))
    quote_data_issue = any(flag in {"STALE", "MISSING_CLOSE", "MISSING_AFTER_HOURS", "MISSING_LATEST"} for flag in quote_flags)
    quote_data_issue = quote_data_issue or quote_missing
    degraded = bool(
        hub_mode == "DEGRADED"
        or not hub_valid
        or chart_data_issue
        or quote_data_issue
        or ("PRICE_MISMATCH" in sanity_flags)
        or bool(chart_error and chart_source in {"cache", "none"})
        or stale
        or provider_down
    )

    grounded = bool(hub_status == "present" and hub_valid and hub_mode == "NORMAL" and not hub_reason)
    degraded_reason = ""
    if hub_mode == "DEGRADED":
        degraded_reason = "Hub is in DEGRADED mode."
    elif not hub_valid:
        degraded_reason = "Hub is invalid."
    elif chart_data_issue:
        if "STALE_CACHE" in chart_flags:
            degraded_reason = "Using cached bars (STALE_CACHE)."
        elif "MISSING" in chart_flags:
            degraded_reason = "Chart data missing."
        else:
            degraded_reason = "Chart quality flags indicate degraded data."
    elif quote_data_issue:
        degraded_reason = "Quote snapshot missing/partial."
    elif "PRICE_MISMATCH" in sanity_flags:
        degraded_reason = "Quote and bars mismatch detected."
    elif chart_error and chart_source in {"cache", "none"}:
        degraded_reason = "Live chart fetch failed; fallback used."
    elif stale:
        degraded_reason = "Overall data is stale."
    elif provider_down:
        degraded_reason = "Market data provider DOWN."
    grounded_reason = "Hub is valid and normal." if grounded else ""

    return {
        "grounded": grounded,
        "degraded": degraded,
        "tool_down": tool_down,
        "stale": stale,
        "grounded_reason": grounded_reason,
        "degraded_reason": degraded_reason,
        "tool_down_reason": tool_down_reason,
    }


def render_badges(trust_state: dict) -> None:
    import streamlit as st

    parts: list[str] = []
    if trust_state.get("grounded"):
        tip = str(trust_state.get("grounded_reason", "Grounded by validated hub.")).strip()
        parts.append(f"<span title='{_escape_attr(tip)}'>✅ Grounded</span>")
    if trust_state.get("degraded"):
        tip = str(trust_state.get("degraded_reason", "Degraded due to data/tool quality.")).strip()
        parts.append(f"<span title='{_escape_attr(tip)}'>⚠️ Degraded</span>")
    if trust_state.get("tool_down"):
        tip = str(trust_state.get("tool_down_reason", "Tool unavailable.")).strip()
        parts.append(f"<span title='{_escape_attr(tip)}'>⛔ Tool Down</span>")
    if trust_state.get("stale"):
        parts.append("<span title='Underlying data is stale.'>🕒 Stale</span>")

    if parts:
        st.markdown(f"<div class='tiny muted'>{' · '.join(parts)}</div>", unsafe_allow_html=True)


def badges_text(trust_state: dict) -> str:
    badges = []
    if trust_state.get("grounded"):
        badges.append("✅")
    if trust_state.get("degraded"):
        badges.append("⚠️")
    if trust_state.get("tool_down"):
        badges.append("⛔")
    if trust_state.get("stale"):
        badges.append("🕒")
    return " ".join(badges)


def _is_provider_failure_error(error_text: str) -> bool:
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


def _extract_channel(context_pack: dict, key: str) -> Any:
    direct = context_pack.get(key)
    if isinstance(direct, dict):
        return direct
    nested = context_pack.get("mcp", {})
    if isinstance(nested, dict):
        channel = nested.get(key)
        if isinstance(channel, dict):
            return channel
    return {}


def _hub_card_citations_valid(hub_card: Any) -> bool:
    if not isinstance(hub_card, dict):
        return False

    used_ids = set(str(cid) for cid in hub_card.get("evidence", {}).get("used_ids", []))
    if not used_ids:
        return False

    for section in ["drivers", "conflicts", "watch"]:
        for item in hub_card.get(section, []):
            citations = item.get("citations", [])
            if not citations:
                return False
            for citation in citations:
                cid = str(citation)
                if not (cid.startswith("indicator:") or cid.startswith("news:") or cid.startswith("macro:")):
                    return False
                if cid not in used_ids:
                    return False

    return True


def _escape_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
