from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from app.core.llm.bedrock_client import BedrockLLMClient, has_aws_credentials
from app.core.llm.hub_card_generator import FORBIDDEN_TERMS, generate_hub_card
from app.core.llm.hub_card_schema import HUB_CARD_JSON_SCHEMA

HUB_FORBIDDEN_TERMS: list[str] = list(dict.fromkeys([*FORBIDDEN_TERMS, "may"]))


class HubValidationFailure(Exception):
    def __init__(self, message: str, candidate_hub: dict | None = None):
        super().__init__(message)
        self.candidate_hub = candidate_hub


@dataclass
class HubGenerationResult:
    status: str
    mode: str
    reason: str | None
    hub_card: dict | None
    cache_path: str | None
    from_cache: bool = False
    hub_valid: bool = False
    llm_usage: dict[str, Any] | None = None


def is_llm_configured(bedrock_config: dict | None = None) -> bool:
    config = resolve_bedrock_config(bedrock_config=bedrock_config)
    has_any_model = bool(config["default_model_id"] or config["claude_model_id"] or config["openai_model_id"])
    if not config["region"] or not has_any_model:
        return False
    return has_aws_credentials()


def resolve_bedrock_config(bedrock_config: dict | None = None) -> dict[str, str]:
    cfg = bedrock_config or {}

    region = (
        str(cfg.get("region", "")).strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )

    default_model = str(cfg.get("model_id", "")).strip() or os.getenv("BEDROCK_MODEL_ID", "").strip()
    claude_model = str(cfg.get("claude_model_id", "")).strip() or os.getenv("BEDROCK_LLM_ID_CLAUDE", "").strip()
    openai_model = str(cfg.get("openai_model_id", "")).strip() or os.getenv("BEDROCK_LLM_ID_OPENAI", "").strip()

    # Allow the generic model ID to serve as fallback for either side.
    if not claude_model:
        claude_model = default_model
    if not openai_model:
        openai_model = default_model

    return {
        "region": region,
        "default_model_id": default_model,
        "claude_model_id": claude_model,
        "openai_model_id": openai_model,
    }


def generate_hub_for_context_pack(
    context_pack: dict,
    now_iso: str,
    bedrock_config: dict | None = None,
    request_timeout_seconds: float | None = None,
) -> HubGenerationResult:
    cfg = resolve_bedrock_config(bedrock_config=bedrock_config)
    ticker = _ticker_from_context_pack(context_pack)
    as_of = _as_of_from_context_pack(context_pack, now_iso=now_iso)
    cache_path = _hub_cache_path(ticker=ticker, as_of_iso=as_of)

    if not is_llm_configured(cfg):
        return HubGenerationResult(
            status="missing",
            mode="DEGRADED",
            reason="LLM not configured",
            hub_card=None,
            cache_path=cache_path,
            from_cache=False,
            hub_valid=False,
            llm_usage=None,
        )

    try:
        hub_card, llm_usage = _generate_conscious_agreement_hub(
            context_pack=context_pack,
            now_iso=now_iso,
            cfg=cfg,
            request_timeout_seconds=request_timeout_seconds,
        )
        _save_cached_hub(cache_path=cache_path, hub_card=hub_card)
        return HubGenerationResult(
            status="present",
            mode="DEGRADED" if hub_card.get("meta", {}).get("mode") == "DEGRADED" else "NORMAL",
            reason=None,
            hub_card=hub_card,
            cache_path=cache_path,
            from_cache=False,
            hub_valid=True,
            llm_usage=llm_usage,
        )
    except Exception as exc:
        dump_path = _debug_dump_rejected_hub(
            ticker=ticker,
            now_iso=now_iso,
            error=str(exc),
            rejected_hub=exc.candidate_hub if isinstance(exc, HubValidationFailure) else None,
        )
        dump_suffix = f" [dump: {dump_path}]" if dump_path else ""

        cached = _load_cached_hub(cache_path=cache_path)
        if isinstance(cached, dict):
            try:
                _validate_hub_card_output(cached, context_pack)
                return HubGenerationResult(
                    status="present",
                    mode="DEGRADED",
                    reason=f"Hub generation failed; cached hub reused ({_one_line(str(exc))}){dump_suffix}",
                    hub_card=cached,
                    cache_path=cache_path,
                    from_cache=True,
                    hub_valid=True,
                    llm_usage=None,
                )
            except Exception:
                pass

        return HubGenerationResult(
            status="invalid",
            mode="DEGRADED",
            reason=f"Hub validation failed ({_one_line(str(exc))}){dump_suffix}",
            hub_card=None,
            cache_path=cache_path,
            from_cache=False,
            hub_valid=False,
            llm_usage=None,
        )


def _generate_conscious_agreement_hub(
    context_pack: dict,
    now_iso: str,
    cfg: dict[str, str],
    request_timeout_seconds: float | None = None,
) -> tuple[dict, dict[str, Any]]:
    region = cfg["region"]
    claude_model = cfg["claude_model_id"]
    openai_model = cfg["openai_model_id"]

    if not claude_model and not openai_model:
        raise RuntimeError("No Bedrock model configured")

    claude_card, claude_usage = _generate_single_model_hub(
        context_pack=context_pack,
        now_iso=now_iso,
        region=region,
        model_id=claude_model or openai_model,
        request_timeout_seconds=request_timeout_seconds,
    )

    if openai_model and openai_model != claude_model:
        openai_card, openai_usage = _generate_single_model_hub(
            context_pack=context_pack,
            now_iso=now_iso,
            region=region,
            model_id=openai_model,
            request_timeout_seconds=request_timeout_seconds,
        )
    else:
        openai_card = claude_card
        openai_usage = {}

    card = _reconcile_cards(
        context_pack=context_pack,
        now_iso=now_iso,
        claude_card=claude_card,
        openai_card=openai_card,
    )
    usage = _merge_llm_usage(
        claude_model=claude_model or openai_model,
        claude_usage=claude_usage,
        openai_model=openai_model or claude_model or "",
        openai_usage=openai_usage,
    )
    return card, usage


def _generate_single_model_hub(
    context_pack: dict,
    now_iso: str,
    region: str,
    model_id: str,
    request_timeout_seconds: float | None = None,
) -> tuple[dict, dict[str, Any]]:
    client = BedrockLLMClient(
        region=region,
        model_id=model_id,
        max_tokens=800,
        temperature=0.0,
        request_timeout_seconds=request_timeout_seconds,
    )
    card = generate_hub_card(context_pack=context_pack, client=client, now_iso=now_iso)
    try:
        _validate_hub_card_output(card, context_pack)
    except Exception as exc:
        raise HubValidationFailure(str(exc), candidate_hub=card) from exc
    return card, client.get_last_usage()


def _merge_llm_usage(
    *,
    claude_model: str,
    claude_usage: dict[str, Any] | None,
    openai_model: str,
    openai_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    first = claude_usage if isinstance(claude_usage, dict) else {}
    second = openai_usage if isinstance(openai_usage, dict) else {}
    first_tokens = first.get("usage", {}) if isinstance(first.get("usage"), dict) else {}
    second_tokens = second.get("usage", {}) if isinstance(second.get("usage"), dict) else {}
    input_tokens = _metric_int(first_tokens, "input_tokens") + _metric_int(second_tokens, "input_tokens")
    output_tokens = _metric_int(first_tokens, "output_tokens") + _metric_int(second_tokens, "output_tokens")
    total_tokens = _metric_int(first_tokens, "total_tokens") + _metric_int(second_tokens, "total_tokens")
    latency_ms = _metric_float(first, "latency_ms") + _metric_float(second, "latency_ms")
    models = {
        "claude": {
            "model_id": str(claude_model or first.get("model_id", "")),
            "transport": str(first.get("transport", "unknown")),
            "latency_ms": round(_metric_float(first, "latency_ms"), 2),
            "usage": {
                "input_tokens": _metric_int(first_tokens, "input_tokens"),
                "output_tokens": _metric_int(first_tokens, "output_tokens"),
                "total_tokens": _metric_int(first_tokens, "total_tokens"),
            },
        },
        "openai": {
            "model_id": str(openai_model or second.get("model_id", "")),
            "transport": str(second.get("transport", "unknown")),
            "latency_ms": round(_metric_float(second, "latency_ms"), 2),
            "usage": {
                "input_tokens": _metric_int(second_tokens, "input_tokens"),
                "output_tokens": _metric_int(second_tokens, "output_tokens"),
                "total_tokens": _metric_int(second_tokens, "total_tokens"),
            },
        },
    }
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens if total_tokens > 0 else input_tokens + output_tokens,
        "latency_ms": round(latency_ms, 2),
        "models": models,
    }


def _reconcile_cards(context_pack: dict, now_iso: str, claude_card: dict, openai_card: dict) -> dict:
    drl_result = context_pack.get("drl", {}).get("result", {})
    action_final = str(drl_result.get("action_final", "WAIT"))
    confidence_cap = float(drl_result.get("confidence_cap", 0.0) or 0.0)

    drivers = _build_deterministic_drivers(context_pack)
    conflicts = _build_deterministic_conflicts(context_pack)
    watch = _build_deterministic_watch(context_pack)

    overlap = _citation_overlap(claude_card.get("drivers", []), openai_card.get("drivers", []))
    consensus_state = "strong" if overlap else "weak"
    summary_line = (
        f"Claude and GPT show {consensus_state} agreement on the DRL {action_final} stance "
        f"with confidence cap {confidence_cap:.0f}."
    )

    reconciled = {
        "meta": {
            "ticker": str(claude_card.get("meta", {}).get("ticker", openai_card.get("meta", {}).get("ticker", ""))),
            "generated_at": now_iso,
            "policy_id": str(claude_card.get("meta", {}).get("policy_id", "")),
            "policy_version": str(claude_card.get("meta", {}).get("policy_version", "")),
            "profile": str(claude_card.get("meta", {}).get("profile", "")),
            "mode": _reconciled_mode(claude_card=claude_card, openai_card=openai_card),
        },
        "summary": {
            "action_final": action_final,
            "confidence_cap": confidence_cap,
            "one_liner": summary_line,
        },
        "drivers": drivers,
        "conflicts": conflicts,
        "watch": watch,
        "evidence": {
            "used_ids": _collect_used_ids(drivers=drivers, conflicts=conflicts, watch=watch),
        },
    }

    reconciled = _postprocess_hub_card(reconciled, context_pack=context_pack)
    _validate_hub_card_output(reconciled, context_pack)
    return reconciled


def _merge_sections(primary: list[Any], secondary: list[Any], min_items: int, max_items: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()

    for raw_item in [*primary, *secondary]:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text", "")).strip()
        citations = [str(c) for c in raw_item.get("citations", []) if str(c).strip()]
        if not text or not citations:
            continue
        if text in seen:
            continue
        merged.append({"text": text, "citations": citations[:4]})
        seen.add(text)
        if len(merged) >= max_items:
            break

    if len(merged) < min_items:
        fallback_id = "indicator:price_last"
        while len(merged) < min_items:
            merged.append(
                {
                    "text": "DRL technical evidence remains the anchor for this view.",
                    "citations": [fallback_id],
                }
            )

    return merged[:max_items]


def _metric(metrics: Any, key: str, default: float) -> float:
    if not isinstance(metrics, dict):
        return float(default)
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _metric_int(metrics: Any, key: str) -> int:
    if not isinstance(metrics, dict):
        return 0
    try:
        return int(float(metrics.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _metric_float(metrics: Any, key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _trend_relation(price_last: float, ema_50: float, sma_200: float) -> str:
    if price_last >= ema_50 and price_last >= sma_200:
        return "above-average"
    if price_last <= ema_50 and price_last <= sma_200:
        return "below-average"
    return "mixed"


def _momentum_state(macd: float, macd_signal: float) -> str:
    if macd > macd_signal:
        return "constructive"
    if macd < macd_signal:
        return "defensive"
    return "neutral"


def _participation_state(adx_14: float, vroc_14: float, atr_pct: float) -> str:
    strength = "strong" if adx_14 >= 25 else "moderate"
    flow = "rising" if vroc_14 >= 20 else ("falling" if vroc_14 <= -10 else "stable")
    vol = "elevated" if atr_pct >= 4.0 else "contained"
    return f"{strength} trend strength with {flow} volume and {vol} volatility"


def _normalize_text(text: str) -> str:
    normalized = text.replace("\\n", " ")
    normalized = " ".join(normalized.split())
    for term in HUB_FORBIDDEN_TERMS:
        normalized = re.sub(rf"\b{re.escape(term)}\b", "", normalized, flags=re.IGNORECASE)
    normalized = " ".join(normalized.split())
    return normalized.strip()


def _has_numeric_anchor(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _build_deterministic_drivers(context_pack: dict) -> list[dict]:
    metrics = context_pack.get("indicators", {}).get("metrics", {}) if isinstance(context_pack, dict) else {}
    price_last = _metric(metrics, "price_last", 0.0)
    ema_50 = _metric(metrics, "ema_50", 0.0)
    sma_200 = _metric(metrics, "sma_200", 0.0)
    rsi_14 = _metric(metrics, "rsi_14", 50.0)
    macd = _metric(metrics, "macd", 0.0)
    macd_signal = _metric(metrics, "macd_signal", 0.0)
    adx_14 = _metric(metrics, "adx_14", 0.0)
    vroc_14 = _metric(metrics, "vroc_14", 0.0)
    atr_pct = _metric(metrics, "atr_pct", 0.0)

    trend_text = (
        f"Trend: price {price_last:.2f}, EMA50 {ema_50:.2f}, and SMA200 {sma_200:.2f} "
        f"place the market in a {_trend_relation(price_last, ema_50, sma_200)} structure."
    )
    momentum_text = (
        f"Momentum: RSI {rsi_14:.1f} and MACD {macd:.2f} vs signal {macd_signal:.2f} "
        f"show a {_momentum_state(macd, macd_signal)} momentum profile."
    )
    participation_text = (
        f"Participation: ADX {adx_14:.1f}, VROC {vroc_14:.1f}, and ATR% {atr_pct:.2f} "
        f"indicate {_participation_state(adx_14, vroc_14, atr_pct)} participation quality."
    )

    return [
        {
            "text": trend_text,
            "citations": ["indicator:price_last", "indicator:ema_50", "indicator:sma_200"],
        },
        {
            "text": momentum_text,
            "citations": ["indicator:rsi_14", "indicator:macd", "indicator:macd_signal"],
        },
        {
            "text": participation_text,
            "citations": ["indicator:adx_14", "indicator:vroc_14", "indicator:atr_pct"],
        },
    ]


def _build_deterministic_conflicts(context_pack: dict) -> list[dict]:
    conflicts = context_pack.get("drl", {}).get("result", {}).get("conflicts", []) if isinstance(context_pack, dict) else []
    metrics = context_pack.get("indicators", {}).get("metrics", {}) if isinstance(context_pack, dict) else {}
    rsi_14 = _metric(metrics, "rsi_14", 50.0)
    atr_pct = _metric(metrics, "atr_pct", 0.0)
    adx_14 = _metric(metrics, "adx_14", 0.0)

    mapped: list[dict] = []
    for code in [str(c) for c in conflicts]:
        if code == "OVERSOLD_BOUNCE_RISK":
            mapped.append(
                {
                    "text": (
                        f"Oversold-bounce risk is active with RSI {rsi_14:.1f}; "
                        "selling into exhaustion remains constrained by policy gates."
                    ),
                    "citations": ["indicator:rsi_14", "indicator:stoch_k"],
                }
            )
        elif code == "HIGH_VOLATILITY":
            mapped.append(
                {
                    "text": f"High-volatility constraint remains active with ATR% at {atr_pct:.2f}.",
                    "citations": ["indicator:atr_pct"],
                }
            )
        elif code == "TIMEFRAME_CONFLICT":
            mapped.append(
                {
                    "text": "Daily and weekly regime alignment is conflicting, reducing directional reliability.",
                    "citations": ["indicator:price_last", "indicator:ema_50", "indicator:sma_200"],
                }
            )
        elif code == "LOW_PARTICIPATION":
            mapped.append(
                {
                    "text": f"Participation is soft with ADX {adx_14:.1f}, reducing conviction.",
                    "citations": ["indicator:adx_14", "indicator:vroc_14"],
                }
            )
    return mapped[:3]


def _build_deterministic_watch(context_pack: dict) -> list[dict]:
    metrics = context_pack.get("indicators", {}).get("metrics", {}) if isinstance(context_pack, dict) else {}
    rsi_14 = _metric(metrics, "rsi_14", 50.0)
    macd = _metric(metrics, "macd", 0.0)
    macd_signal = _metric(metrics, "macd_signal", 0.0)

    return [
        {
            "text": f"Watch RSI around 45/65 transition bands; current RSI is {rsi_14:.1f}.",
            "citations": ["indicator:rsi_14"],
        },
        {
            "text": f"Watch MACD {macd:.2f} versus signal {macd_signal:.2f} for momentum confirmation.",
            "citations": ["indicator:macd", "indicator:macd_signal"],
        },
    ]


def _postprocess_hub_card(card: dict, context_pack: dict) -> dict:
    processed = json.loads(json.dumps(card))
    if not isinstance(processed.get("evidence"), dict):
        processed["evidence"] = {"used_ids": []}
    has_news = len(_channel_items(context_pack, key="news")) > 0
    has_macro = len(_channel_items(context_pack, key="macro")) > 0
    indicators_only = not has_news and not has_macro

    for section in ["drivers", "conflicts", "watch"]:
        items = processed.get(section, [])
        if not isinstance(items, list):
            continue
        deduped: list[dict] = []
        seen: set[str] = set()
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            text = _normalize_text(str(raw_item.get("text", "")))
            citations = [str(c).strip() for c in raw_item.get("citations", []) if str(c).strip()]
            if indicators_only:
                citations = [cid for cid in citations if cid.startswith("indicator:")]
            if not citations:
                citations = ["indicator:price_last"]
            signature = f"{text}|{'|'.join(citations)}"
            if not text or signature in seen:
                continue
            seen.add(signature)
            deduped.append({"text": text, "citations": citations[:4]})
        processed[section] = deduped

    # Ensure driver cardinality and numeric anchors.
    drivers = processed.get("drivers", [])
    if not isinstance(drivers, list):
        drivers = []
    drivers = drivers[:3]
    while len(drivers) < 3:
        drivers.append(
            {
                "text": "Trend and momentum remain governed by current indicator readings.",
                "citations": ["indicator:price_last"],
            }
        )
    for idx, item in enumerate(drivers):
        text = str(item.get("text", ""))
        if not _has_numeric_anchor(text):
            text = f"{text} Current price_last is {_metric(context_pack.get('indicators', {}).get('metrics', {}), 'price_last', 0.0):.2f}."
        drivers[idx] = {"text": text, "citations": item.get("citations", ["indicator:price_last"])}
    processed["drivers"] = drivers

    # Recompute used IDs to prevent concatenation issues.
    processed["evidence"]["used_ids"] = _collect_used_ids(
        drivers=processed.get("drivers", []),
        conflicts=processed.get("conflicts", []),
        watch=processed.get("watch", []),
    )
    return processed


def _citation_overlap(section_a: list[Any], section_b: list[Any]) -> set[str]:
    citations_a: set[str] = set()
    citations_b: set[str] = set()

    for item in section_a:
        if isinstance(item, dict):
            citations_a.update(str(c) for c in item.get("citations", []))
    for item in section_b:
        if isinstance(item, dict):
            citations_b.update(str(c) for c in item.get("citations", []))

    return citations_a & citations_b


def _reconciled_mode(claude_card: dict, openai_card: dict) -> str:
    modes = {
        str(claude_card.get("meta", {}).get("mode", "")),
        str(openai_card.get("meta", {}).get("mode", "")),
    }
    if "DEGRADED" in modes:
        return "DEGRADED"
    if "FULL" in modes:
        return "FULL"
    return "TECHNICAL_ONLY"


def _validate_hub_card_output(card: dict, context_pack: dict) -> None:
    try:
        validate(instance=card, schema=HUB_CARD_JSON_SCHEMA)
    except ValidationError as exc:
        raise ValueError(f"HUB_SCHEMA_INVALID: {exc.message}") from exc

    drl = context_pack.get("drl", {}).get("result", {})
    expected_action = str(drl.get("action_final", ""))
    expected_conf = float(drl.get("confidence_cap", 0.0) or 0.0)

    if str(card.get("summary", {}).get("action_final", "")) != expected_action:
        raise ValueError("HUB_ACTION_MISMATCH")

    actual_conf = float(card.get("summary", {}).get("confidence_cap", 0.0) or 0.0)
    if abs(actual_conf - expected_conf) > 0.1:
        raise ValueError("HUB_CONFIDENCE_MISMATCH")

    allowed_ids = _allowed_evidence_ids(context_pack)
    if not allowed_ids:
        raise ValueError("HUB_NO_EVIDENCE")

    used_ids = {str(cid) for cid in card.get("evidence", {}).get("used_ids", [])}
    cited_ids: set[str] = set()
    for section in ["drivers", "conflicts", "watch"]:
        for item in card.get(section, []):
            if not isinstance(item, dict):
                raise ValueError("HUB_SECTION_ITEM_INVALID")
            text = str(item.get("text", ""))
            forbidden_term = _matched_forbidden_term(text)
            if forbidden_term:
                raise ValueError(f"HUB_FORBIDDEN_TERM:{forbidden_term}")
            citations = [str(c) for c in item.get("citations", [])]
            if not citations:
                raise ValueError("HUB_MISSING_CITATIONS")
            for cid in citations:
                if not (cid.startswith("indicator:") or cid.startswith("news:") or cid.startswith("macro:")):
                    raise ValueError("HUB_INVALID_CITATION_PREFIX")
                if cid not in allowed_ids:
                    raise ValueError("HUB_UNKNOWN_CITATION")
                cited_ids.add(cid)

    if not cited_ids.issubset(used_ids):
        raise ValueError("HUB_USED_IDS_MISMATCH")
    if not used_ids.issubset(allowed_ids):
        raise ValueError("HUB_USED_IDS_UNKNOWN")


def _allowed_evidence_ids(context_pack: dict) -> set[str]:
    allowed: set[str] = set()

    metrics = context_pack.get("indicators", {}).get("metrics", {})
    if isinstance(metrics, dict):
        for name in metrics.keys():
            allowed.add(f"indicator:{name}")

    for item in _channel_items(context_pack, key="news"):
        item_id = str(item.get("id", "")).strip()
        if item_id:
            allowed.add(f"news:{item_id}")

    for item in _channel_items(context_pack, key="macro"):
        item_id = str(item.get("id", "")).strip()
        if item_id:
            allowed.add(f"macro:{item_id}")

    return allowed


def _channel_items(context_pack: dict, key: str) -> list[dict]:
    direct = context_pack.get(key)
    if isinstance(direct, dict):
        items = direct.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    nested = context_pack.get("mcp", {})
    if isinstance(nested, dict):
        channel = nested.get(key)
        if isinstance(channel, dict):
            items = channel.get("items", [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]

    return []


def _collect_used_ids(drivers: list[dict], conflicts: list[dict], watch: list[dict]) -> list[str]:
    used: list[str] = []
    for section in [drivers, conflicts, watch]:
        for item in section:
            if not isinstance(item, dict):
                continue
            for citation in item.get("citations", []):
                cid = str(citation)
                if cid and cid not in used:
                    used.append(cid)
    return used[:30]


def _matched_forbidden_term(text: str) -> str | None:
    lower = text.lower()
    for term in HUB_FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lower):
            return str(term)
    return None


def _hub_cache_path(ticker: str, as_of_iso: str) -> str:
    safe_ticker = re.sub(r"[^A-Z0-9._-]", "_", ticker.upper()) or "UNKNOWN"
    safe_as_of = re.sub(r"[^0-9A-Za-z]", "", as_of_iso)[:24] or "unknown"
    path = Path(".cache") / "hub_cards" / f"{safe_ticker}-{safe_as_of}.json"
    return str(path)


def _save_cached_hub(cache_path: str, hub_card: dict) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hub_card, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _load_cached_hub(cache_path: str) -> dict | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ticker_from_context_pack(context_pack: dict) -> str:
    ticker = str(context_pack.get("meta", {}).get("ticker", "")).strip().upper()
    if ticker:
        return ticker
    return str(context_pack.get("drl", {}).get("result", {}).get("decision_trace", {}).get("ticker", "UNKNOWN")).strip().upper() or "UNKNOWN"


def _as_of_from_context_pack(context_pack: dict, now_iso: str) -> str:
    as_of = str(context_pack.get("prices", {}).get("as_of", "")).strip()
    return as_of or now_iso


def _one_line(text: str) -> str:
    return " ".join(text.split())[:180]


def _debug_dump_rejected_hub(
    ticker: str,
    now_iso: str,
    error: str,
    rejected_hub: dict | None,
) -> str | None:
    if os.getenv("HUB_DEBUG_DUMP", "").strip() != "1":
        return None

    safe_ticker = re.sub(r"[^A-Z0-9._-]", "_", str(ticker).upper()) or "UNKNOWN"
    safe_ts = re.sub(r"[^0-9A-Za-z]", "", str(now_iso))[:24] or "unknown"
    path = Path(".cache") / "hub_rejected" / f"{safe_ticker}_{safe_ts}.json"
    payload = {
        "ticker": safe_ticker,
        "timestamp": now_iso,
        "error": _one_line(error),
        "hub_card": rejected_hub if isinstance(rejected_hub, dict) else None,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        return str(path)
    except OSError:
        return None
