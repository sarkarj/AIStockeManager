from __future__ import annotations

import re
from typing import Any

from jsonschema import ValidationError, validate

from app.core.llm.hub_card_schema import HUB_CARD_JSON_SCHEMA

FORBIDDEN_TERMS = ["could", "might", "may", "potentially", "possibly", "speculate"]


def generate_hub_card(context_pack: dict, client: Any, now_iso: str) -> dict:
    mode = _determine_mode(context_pack)
    evidence_catalog = _collect_evidence_catalog(context_pack)

    for retry_idx in range(2):
        prompt = _build_prompt(context_pack, evidence_catalog, mode=mode, stricter=(retry_idx == 1))
        try:
            candidate = client.invoke_structured(prompt=prompt, json_schema=HUB_CARD_JSON_SCHEMA)
        except Exception:
            continue

        try:
            _validate_hub_card(candidate, context_pack, evidence_catalog, expected_mode=mode)
            return candidate
        except Exception:
            continue

    fallback_mode = "DEGRADED" if mode == "DEGRADED" else "TECHNICAL_ONLY"
    return build_fallback_hub_card(context_pack=context_pack, now_iso=now_iso, mode_override=fallback_mode)


def build_fallback_hub_card(context_pack: dict, now_iso: str, mode_override: str | None = None) -> dict:
    evidence_catalog = _collect_evidence_catalog(context_pack)
    indicator_ids = sorted(evidence_catalog["indicator_ids"])

    def pick_indicator(name: str) -> str:
        preferred = f"indicator:{name}"
        if preferred in evidence_catalog["allowed_ids"]:
            return preferred
        if indicator_ids:
            return indicator_ids[0]
        return preferred

    drl = context_pack.get("drl", {}).get("result", {})
    trace = drl.get("decision_trace", {})
    metrics = context_pack.get("indicators", {}).get("metrics", {})

    action_final = str(drl.get("action_final", "WAIT"))
    confidence_cap = float(drl.get("confidence_cap", 0))

    price_last = float(metrics.get("price_last", 0.0))
    ema_50 = float(metrics.get("ema_50", 0.0))
    rsi_14 = float(metrics.get("rsi_14", 50.0))
    macd = float(metrics.get("macd", 0.0))
    macd_signal = float(metrics.get("macd_signal", 0.0))

    ema_relation = "ABOVE" if price_last > ema_50 else "BELOW"
    macd_state = "BULL" if macd > macd_signal else "BEAR"

    mode = mode_override or _determine_mode(context_pack)

    drivers = [
        {
            "text": f"Price is {ema_relation} EMA50, framing the short-term technical posture.",
            "citations": [pick_indicator("price_last"), pick_indicator("ema_50")],
        },
        {
            "text": f"Momentum is {macd_state} with RSI at {rsi_14:.1f}, shaping near-term conviction.",
            "citations": [pick_indicator("macd"), pick_indicator("macd_signal"), pick_indicator("rsi_14")],
        },
    ]

    conflicts = []
    for code in drl.get("conflicts", []):
        text, cits = _fallback_conflict_from_code(str(code), pick_indicator)
        conflicts.append({"text": text, "citations": cits})

    watch = [
        {
            "text": "Watch RSI movement across policy zones to confirm momentum persistence.",
            "citations": [pick_indicator("rsi_14")],
        },
        {
            "text": "Watch MACD versus signal direction for confirmation or reversal risk.",
            "citations": [pick_indicator("macd"), pick_indicator("macd_signal")],
        },
    ]

    used_ids = _collect_used_ids(drivers=drivers, conflicts=conflicts, watch=watch)
    if len(used_ids) < 2:
        used_ids = sorted(list(set(used_ids + [pick_indicator("price_last"), pick_indicator("ema_50")])))[0:30]

    card = {
        "meta": {
            "ticker": str(context_pack.get("meta", {}).get("ticker", trace.get("ticker", ""))),
            "generated_at": now_iso,
            "policy_id": str(trace.get("policy_id", "")),
            "policy_version": str(trace.get("policy_version", "")),
            "profile": str(trace.get("profile", "")),
            "mode": mode,
        },
        "summary": {
            "action_final": action_final,
            "confidence_cap": confidence_cap,
            "one_liner": (
                f"DRL holds {action_final} at confidence cap {confidence_cap:.0f}; RSI {rsi_14:.1f}, "
                f"MACD {macd_state}, and price {ema_relation} EMA50 drive the current view."
            ),
        },
        "drivers": drivers,
        "conflicts": conflicts,
        "watch": watch,
        "evidence": {
            "used_ids": used_ids,
        },
    }

    # Fallback itself must be strict and grounded.
    _validate_hub_card(card, context_pack, evidence_catalog, expected_mode=mode)
    return card


def _determine_mode(context_pack: dict) -> str:
    overall_stale = bool(context_pack.get("meta", {}).get("data_quality", {}).get("overall_stale", False))
    tool_down = _has_tool_down_marker(context_pack)
    has_news = len(_extract_news_items(context_pack)) > 0
    has_macro = len(_extract_macro_items(context_pack)) > 0

    if overall_stale or tool_down:
        return "DEGRADED"
    if has_news or has_macro:
        return "FULL"
    return "TECHNICAL_ONLY"


def _has_tool_down_marker(value: Any) -> bool:
    if isinstance(value, dict):
        err = value.get("error")
        if isinstance(err, dict):
            code = str(err.get("code", ""))
            if code in {"TOOL_DOWN", "BEDROCK_UNAVAILABLE", "STALE_DATA"}:
                return True
        for v in value.values():
            if _has_tool_down_marker(v):
                return True
    elif isinstance(value, list):
        for v in value:
            if _has_tool_down_marker(v):
                return True
    elif isinstance(value, str):
        upper = value.upper()
        if "TOOL_DOWN" in upper or "BEDROCK_UNAVAILABLE" in upper:
            return True
    return False


def _extract_news_items(context_pack: dict) -> list[dict]:
    candidates = []
    if isinstance(context_pack.get("news"), dict):
        candidates.extend(context_pack.get("news", {}).get("items", []))
    if isinstance(context_pack.get("mcp"), dict) and isinstance(context_pack["mcp"].get("news"), dict):
        candidates.extend(context_pack["mcp"]["news"].get("items", []))
    return [item for item in candidates if isinstance(item, dict)]


def _extract_macro_items(context_pack: dict) -> list[dict]:
    candidates = []
    if isinstance(context_pack.get("macro"), dict):
        candidates.extend(context_pack.get("macro", {}).get("items", []))
    if isinstance(context_pack.get("mcp"), dict) and isinstance(context_pack["mcp"].get("macro"), dict):
        candidates.extend(context_pack["mcp"]["macro"].get("items", []))
    return [item for item in candidates if isinstance(item, dict)]


def _collect_evidence_catalog(context_pack: dict) -> dict:
    metrics = context_pack.get("indicators", {}).get("metrics", {})
    indicator_ids = {f"indicator:{name}" for name in metrics.keys()}

    news_ids = set()
    for item in _extract_news_items(context_pack):
        if "id" in item:
            news_ids.add(f"news:{item['id']}")

    macro_ids = set()
    for item in _extract_macro_items(context_pack):
        if "id" in item:
            macro_ids.add(f"macro:{item['id']}")

    allowed_ids = set().union(indicator_ids, news_ids, macro_ids)
    return {
        "indicator_ids": indicator_ids,
        "news_ids": news_ids,
        "macro_ids": macro_ids,
        "allowed_ids": allowed_ids,
    }


def _build_prompt(context_pack: dict, evidence_catalog: dict, mode: str, stricter: bool) -> str:
    drl = context_pack.get("drl", {}).get("result", {})
    metrics = context_pack.get("indicators", {}).get("metrics", {})

    news_items = _extract_news_items(context_pack)
    macro_items = _extract_macro_items(context_pack)

    indicator_lines = [f"- indicator:{k} = {metrics[k]}" for k in sorted(metrics.keys())]
    news_lines = [
        f"- news:{item.get('id')} | {item.get('published_at','')} | {item.get('title','')}"
        for item in news_items
        if item.get("id")
    ]
    macro_lines = [
        f"- macro:{item.get('id')} | {item.get('label','')} = {item.get('value','')}"
        for item in macro_items
        if item.get("id")
    ]

    extra_rule = "Reject any text containing forbidden words." if stricter else ""

    return "\n".join(
        [
            "Create an Intelligence Hub Card as strict JSON.",
            "Rules:",
            f"- action_final MUST equal DRL action_final: {drl.get('action_final')}",
            f"- confidence_cap MUST equal DRL confidence_cap: {drl.get('confidence_cap')}",
            f"- mode MUST be: {mode}",
            "- Every drivers/conflicts/watch item must include citations with IDs from the provided evidence catalog.",
            "- Allowed citation formats: indicator:<name>, news:<id>, macro:<id>.",
            "- No hedge words: could, might, potentially, possibly, speculate.",
            "- Output JSON only matching schema.",
            extra_rule,
            "",
            "DRL context:",
            f"- action_final={drl.get('action_final')}",
            f"- confidence_cap={drl.get('confidence_cap')}",
            f"- conflicts={drl.get('conflicts', [])}",
            "",
            "Indicator evidence:",
            *indicator_lines,
            "",
            "News evidence:",
            *(news_lines if news_lines else ["- none"]),
            "",
            "Macro evidence:",
            *(macro_lines if macro_lines else ["- none"]),
            "",
            f"Allowed evidence IDs: {sorted(evidence_catalog['allowed_ids'])}",
        ]
    )


def _validate_hub_card(card: dict, context_pack: dict, evidence_catalog: dict, expected_mode: str) -> None:
    try:
        validate(instance=card, schema=HUB_CARD_JSON_SCHEMA)
    except ValidationError as exc:
        raise ValueError(f"SCHEMA_INVALID: {exc.message}") from exc

    drl = context_pack.get("drl", {}).get("result", {})
    expected_action = str(drl.get("action_final", ""))
    expected_conf = float(drl.get("confidence_cap", 0.0))

    if str(card["summary"]["action_final"]) != expected_action:
        raise ValueError("ACTION_MISMATCH")

    actual_conf = float(card["summary"]["confidence_cap"])
    if abs(actual_conf - expected_conf) > 0.1:
        raise ValueError("CONFIDENCE_MISMATCH")

    if card["meta"]["mode"] != expected_mode:
        raise ValueError("MODE_MISMATCH")

    _validate_citations(card, evidence_catalog)
    _validate_forbidden_terms(card)


def _validate_citations(card: dict, evidence_catalog: dict) -> None:
    allowed_ids = evidence_catalog["allowed_ids"]
    if not allowed_ids:
        raise ValueError("NO_EVIDENCE_AVAILABLE")

    cited: set[str] = set()
    for section in ["drivers", "conflicts", "watch"]:
        for item in card.get(section, []):
            citations = item.get("citations", [])
            if not citations:
                raise ValueError("MISSING_CITATION")
            for c in citations:
                cid = str(c)
                if not (cid.startswith("indicator:") or cid.startswith("news:") or cid.startswith("macro:")):
                    raise ValueError("INVALID_CITATION_PREFIX")
                if cid not in allowed_ids:
                    raise ValueError("CITATION_NOT_IN_EVIDENCE")
                cited.add(cid)

    used_ids = set(str(i) for i in card.get("evidence", {}).get("used_ids", []))
    if not cited.issubset(used_ids):
        raise ValueError("USED_IDS_MISSING_CITED")

    if not used_ids.issubset(allowed_ids):
        raise ValueError("USED_IDS_NOT_IN_EVIDENCE")


def _validate_forbidden_terms(card: dict) -> None:
    text_values = [
        str(card.get("summary", {}).get("one_liner", "")),
    ]
    for section in ["drivers", "conflicts", "watch"]:
        for item in card.get(section, []):
            text_values.append(str(item.get("text", "")))

    for text in text_values:
        lower_text = text.lower()
        for term in FORBIDDEN_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lower_text):
                raise ValueError("FORBIDDEN_TERM")


def _fallback_conflict_from_code(code: str, pick_indicator: Any) -> tuple[str, list[str]]:
    mapping = {
        "OVERSOLD_BOUNCE_RISK": (
            "Oversold conditions reduce confidence for additional downside follow-through.",
            [pick_indicator("rsi_14"), pick_indicator("stoch_k")],
        ),
        "OVERBOUGHT_PULLBACK_RISK": (
            "Overbought readings increase pullback risk and reduce chase quality.",
            [pick_indicator("rsi_14"), pick_indicator("stoch_k")],
        ),
        "TIMEFRAME_CONFLICT": (
            "Timeframe alignment is mixed, reducing directional reliability.",
            [pick_indicator("ema_50"), pick_indicator("sma_200")],
        ),
        "HIGH_VOLATILITY": (
            "Elevated ATR indicates higher path risk and lower conviction.",
            [pick_indicator("atr_pct")],
        ),
        "LOW_PARTICIPATION": (
            "Weak participation lowers conviction behind the current move.",
            [pick_indicator("vroc_14")],
        ),
        "STALE_DATA": (
            "Input data freshness is below threshold and confidence is capped.",
            [pick_indicator("price_last")],
        ),
    }
    return mapping.get(
        code,
        (
            "Risk controls remain active under current technical conditions.",
            [pick_indicator("price_last")],
        ),
    )


def _collect_used_ids(drivers: list[dict], conflicts: list[dict], watch: list[dict]) -> list[str]:
    ids = set()
    for section in [drivers, conflicts, watch]:
        for item in section:
            for cid in item.get("citations", []):
                ids.add(str(cid))
    return sorted(ids)[0:30]
