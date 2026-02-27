from __future__ import annotations

import re
from typing import Any

from app.core.context_pack.hub_generator import HUB_FORBIDDEN_TERMS

_CONCATENATED_INDICATOR_PATTERN = re.compile(r"indicator:[a-z0-9_]+indicator:", re.IGNORECASE)


def verify_hub_integrity(hub: dict, indicators: dict | None = None) -> dict:
    h1 = check_h1_forbidden_terms(hub)
    h2 = check_h2_citation_format_and_dedupe(hub)
    h3 = check_h3_numeric_contradictions(hub, indicators=indicators)

    violations: list[dict] = []
    violations.extend(h1["violations"])
    violations.extend(h2["violations"])
    violations.extend(h3["violations"])

    skipped_rules: list[str] = []
    if h3.get("skipped", False):
        skipped_rules.append("H3")

    return {
        "ok": bool(h1["ok"] and h2["ok"] and h3["ok"]),
        "violations": violations,
        "rules": {
            "H1": h1,
            "H2": h2,
            "H3": h3,
        },
        "canonical_citations": h2.get("canonical_citations", []),
        "skipped_rules": skipped_rules,
    }


def check_h1_forbidden_terms(hub_text_fields: dict) -> dict:
    violations: list[dict] = []
    texts = _extract_narrative_texts(hub_text_fields)
    for text in texts:
        lowered = text.lower()
        for token in HUB_FORBIDDEN_TERMS:
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                violations.append(
                    {
                        "rule_id": "H1",
                        "snippet": _truncate(text),
                        "expected_relation": f"no forbidden token '{token}'",
                        "actual_values": {"token": token},
                    }
                )
    return {"ok": len(violations) == 0, "violations": violations}


def check_h2_citation_format_and_dedupe(hub: dict) -> dict:
    violations: list[dict] = []
    raw_tokens = _extract_citation_tokens(hub)
    canonical: list[str] = []
    seen: set[str] = set()

    for raw in raw_tokens:
        text = str(raw).strip()
        if not text:
            continue

        if "Refs: Citations:" in text:
            violations.append(
                {
                    "rule_id": "H2",
                    "snippet": _truncate(text),
                    "expected_relation": "must not contain literal 'Refs: Citations:'",
                    "actual_values": {"token": text},
                }
            )

        parts = _split_and_normalize_refs(text)
        for token in parts:
            if _CONCATENATED_INDICATOR_PATTERN.search(token):
                violations.append(
                    {
                        "rule_id": "H2",
                        "snippet": _truncate(token),
                        "expected_relation": "no concatenated indicator tokens",
                        "actual_values": {"token": token},
                    }
                )

            if any(ch.isspace() for ch in token):
                violations.append(
                    {
                        "rule_id": "H2",
                        "snippet": _truncate(token),
                        "expected_relation": "citation token must not contain whitespace/newline",
                        "actual_values": {"token": token},
                    }
                )
                continue

            if token in seen:
                violations.append(
                    {
                        "rule_id": "H2",
                        "snippet": _truncate(token),
                        "expected_relation": "deduplicated citations",
                        "actual_values": {"token": token},
                    }
                )
                continue

            seen.add(token)
            canonical.append(token)

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "canonical_citations": canonical,
    }


def check_h3_numeric_contradictions(hub_text: dict, indicators: dict | None) -> dict:
    if not isinstance(indicators, dict) or not indicators:
        return {"ok": True, "violations": [], "skipped": True}

    violations: list[dict] = []
    texts = _extract_narrative_texts(hub_text)
    metrics = _normalize_metrics(indicators)

    rules = [
        (
            r"\bprice\s+above\s+ema\s*50\b|\bprice\s+above\s+ema50\b",
            "price_last > ema_50",
            lambda m: _is_true(m["price_last"] > m["ema_50"]),
            ["price_last", "ema_50"],
        ),
        (
            r"\bema\s*50\s+above\s+price\b|\bema50\s+above\s+price\b",
            "ema_50 > price_last",
            lambda m: _is_true(m["ema_50"] > m["price_last"]),
            ["ema_50", "price_last"],
        ),
        (
            r"\bprice\s+above\s+sma\s*200\b|\bprice\s+above\s+sma200\b",
            "price_last > sma_200",
            lambda m: _is_true(m["price_last"] > m["sma_200"]),
            ["price_last", "sma_200"],
        ),
        (
            r"\bsma\s*200\s+above\s+price\b|\bsma200\s+above\s+price\b",
            "sma_200 > price_last",
            lambda m: _is_true(m["sma_200"] > m["price_last"]),
            ["sma_200", "price_last"],
        ),
        (
            r"\brsi\s+oversold\b",
            "rsi_14 < 30",
            lambda m: _is_true(m["rsi_14"] < 30),
            ["rsi_14"],
        ),
        (
            r"\brsi\s+overbought\b",
            "rsi_14 > 70",
            lambda m: _is_true(m["rsi_14"] > 70),
            ["rsi_14"],
        ),
        (
            r"\bmacd\s+bullish\b",
            "macd > macd_signal",
            lambda m: _is_true(m["macd"] > m["macd_signal"]),
            ["macd", "macd_signal"],
        ),
        (
            r"\bmacd\s+bearish\b",
            "macd < macd_signal",
            lambda m: _is_true(m["macd"] < m["macd_signal"]),
            ["macd", "macd_signal"],
        ),
        (
            r"\badx\s+strong\s+trend\b",
            "adx_14 >= 25",
            lambda m: _is_true(m["adx_14"] >= 25),
            ["adx_14"],
        ),
        (
            r"\badx\s+weak\s+trend\b",
            "adx_14 < 20",
            lambda m: _is_true(m["adx_14"] < 20),
            ["adx_14"],
        ),
    ]

    for text in texts:
        lowered = text.lower()
        for pattern, expected_relation, assertion, keys in rules:
            if re.search(pattern, lowered):
                if not assertion(metrics):
                    violations.append(
                        {
                            "rule_id": "H3",
                            "snippet": _truncate(text),
                            "expected_relation": expected_relation,
                            "actual_values": {key: metrics.get(key) for key in keys},
                        }
                    )

    return {"ok": len(violations) == 0, "violations": violations, "skipped": False}


# Backward-compatible aliases from prior internal draft.
def check_h1_hedge_words(hub: dict) -> dict:
    return check_h1_forbidden_terms(hub)


def check_h2_citation_formatting(hub: dict) -> dict:
    return check_h2_citation_format_and_dedupe(hub)


def _extract_narrative_texts(node: Any) -> list[str]:
    texts: list[str] = []
    _collect_narrative_texts(node, texts)
    return texts


def _collect_narrative_texts(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lower_key = str(key).lower()
            if lower_key in {"used_ids", "id", "url", "source", "cache_path"}:
                continue
            if isinstance(value, str):
                if _is_narrative_key(lower_key) or _looks_like_sentence(value):
                    out.append(value)
            else:
                _collect_narrative_texts(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_narrative_texts(item, out)


def _extract_citation_tokens(node: Any) -> list[str]:
    tokens: list[str] = []
    _collect_citation_tokens(node, tokens)
    return tokens


def _collect_citation_tokens(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lower_key = str(key).lower()
            if lower_key in {"citations", "refs", "references", "contentreferences", "content_references"}:
                if isinstance(value, str):
                    out.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            out.append(item)
                        elif isinstance(item, dict):
                            ref_id = item.get("id") or item.get("ref") or item.get("value")
                            if ref_id is not None:
                                out.append(str(ref_id))
                continue
            _collect_citation_tokens(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_citation_tokens(item, out)


def _split_and_normalize_refs(text: str) -> list[str]:
    cleaned = text.strip()
    while True:
        updated = re.sub(r"(?i)^\s*(refs|citations)\s*:\s*", "", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated
    if not cleaned:
        return []
    parts = [part.strip() for part in cleaned.split(",")]
    return [part.lower() for part in parts if part]


def _normalize_metrics(indicators: dict[str, Any]) -> dict[str, float]:
    keys = ["price_last", "ema_50", "sma_200", "rsi_14", "macd", "macd_signal", "adx_14"]
    return {key: _to_float(indicators.get(key)) for key in keys}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_true(value: bool) -> bool:
    return bool(value)


def _is_narrative_key(key: str) -> bool:
    return key in {
        "text",
        "one_liner",
        "consensus_summary",
        "summary",
        "narrative",
        "description",
        "reason",
        "drivers",
        "risks",
        "watch",
        "conflicts",
        "citations",
        "refs",
        "references",
    }


def _looks_like_sentence(text: str) -> bool:
    compact = " ".join(str(text).split())
    return len(compact) >= 12 and " " in compact


def _truncate(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
