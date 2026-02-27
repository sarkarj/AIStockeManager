from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

WHY_CACHE_DIR = Path(".cache") / "why"
WHY_SIGNATURE_VERSION = "why-v1"
WHY_PROMPT_VERSION = "hub-prompt-v1"
WHY_VALIDATOR_VERSION = "hub-validator-v1"


@dataclass(frozen=True)
class WhyArtifact:
    signature: str
    ticker: str
    generated_at: str
    hub_card: dict[str, Any]
    hub_meta: dict[str, Any]
    source: str = "cache"


def build_why_signature(
    *,
    ticker: str,
    drl_result: dict[str, Any],
    indicators: dict[str, Any],
    quote: dict[str, Any],
    range_key: str,
    model_fingerprint: str | None = None,
    prompt_version: str = WHY_PROMPT_VERSION,
    validator_version: str = WHY_VALIDATOR_VERSION,
) -> str:
    metrics = indicators.get("metrics", {}) if isinstance(indicators, dict) else {}
    payload = {
        "signature_version": WHY_SIGNATURE_VERSION,
        "prompt_version": str(prompt_version),
        "validator_version": str(validator_version),
        "model_fingerprint": str(model_fingerprint or _default_model_fingerprint()),
        "ticker": str(ticker or "").strip().upper(),
        "range_key": str(range_key or "1D").strip().upper(),
        "action_final": drl_result.get("action_final"),
        "confidence_cap": drl_result.get("confidence_cap"),
        "gates_triggered": drl_result.get("gates_triggered", []),
        "conflicts": drl_result.get("conflicts", []),
        "indicator_as_of": indicators.get("as_of") if isinstance(indicators, dict) else None,
        "indicator_metrics": metrics if isinstance(metrics, dict) else {},
        "quote": {
            "latest_price": quote.get("latest_price"),
            "close_price": quote.get("close_price"),
            "after_hours_price": quote.get("after_hours_price"),
            "prev_close_price": quote.get("prev_close_price"),
            "latest_ts": quote.get("latest_ts"),
            "close_ts": quote.get("close_ts"),
            "after_hours_ts": quote.get("after_hours_ts"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def load_why_artifact(*, signature: str, ticker: str | None = None) -> WhyArtifact | None:
    sig = str(signature or "").strip().lower()
    if not sig:
        return None
    path = _artifact_path(signature=sig, ticker=ticker)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("signature", "")).strip().lower() != sig:
        return None
    hub_card = payload.get("hub_card", {})
    hub_meta = payload.get("hub_meta", {})
    ticker_value = str(payload.get("ticker", "")).strip().upper()
    if not ticker_value or not isinstance(hub_card, dict):
        return None
    if not isinstance(hub_meta, dict):
        hub_meta = {}
    return WhyArtifact(
        signature=sig,
        ticker=ticker_value,
        generated_at=str(payload.get("generated_at", "")).strip(),
        hub_card=hub_card,
        hub_meta=hub_meta,
        source=str(payload.get("source", "cache") or "cache"),
    )


def load_latest_why_artifact(*, ticker: str, max_age_minutes: int | None = None) -> WhyArtifact | None:
    symbol = str(ticker or "").strip().upper()
    if not symbol or not WHY_CACHE_DIR.exists():
        return None
    pattern = f"{symbol}-*.json"
    candidates = sorted(WHY_CACHE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        signature = str(payload.get("signature", "")).strip().lower()
        if not signature:
            continue
        artifact = load_why_artifact(signature=signature, ticker=symbol)
        if artifact is None:
            continue
        if isinstance(max_age_minutes, int) and max_age_minutes > 0:
            generated_at = _parse_iso_or_none(artifact.generated_at)
            if generated_at is None:
                continue
            age_minutes = (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds() / 60.0
            if age_minutes > float(max_age_minutes):
                continue
        return artifact
    return None


def save_why_artifact(
    *,
    signature: str,
    ticker: str,
    hub_card: dict[str, Any],
    hub_meta: dict[str, Any] | None = None,
    generated_at: str,
) -> str:
    sig = str(signature or "").strip().lower()
    symbol = str(ticker or "").strip().upper()
    if not sig or not symbol or not isinstance(hub_card, dict):
        return ""
    WHY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(signature=sig, ticker=symbol)
    payload = {
        "signature": sig,
        "ticker": symbol,
        "generated_at": str(generated_at or ""),
        "hub_card": hub_card,
        "hub_meta": hub_meta if isinstance(hub_meta, dict) else {},
        "source": "cache",
    }
    _atomic_write_json(path=path, payload=payload)
    return str(path)


def hydrate_context_pack_with_why(context_pack: dict[str, Any], artifact: WhyArtifact) -> dict[str, Any]:
    pack = context_pack if isinstance(context_pack, dict) else {}
    pack["hub_card"] = artifact.hub_card
    meta = pack.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        pack["meta"] = meta
    hub_meta = dict(artifact.hub_meta or {})
    hub_meta.setdefault("status", "present")
    hub_meta.setdefault("mode", "NORMAL")
    hub_meta.setdefault("reason", None)
    hub_meta.setdefault("hub_valid", True)
    hub_meta["why_signature"] = artifact.signature
    hub_meta["why_source"] = artifact.source
    hub_meta["why_cached_at"] = artifact.generated_at
    meta["hub"] = hub_meta
    return pack


def _artifact_path(*, signature: str, ticker: str | None = None) -> Path:
    symbol = str(ticker or "TICKER").strip().upper() or "TICKER"
    safe = "".join(ch for ch in symbol if ch.isalnum() or ch in {"-", "_", "."})[:16] or "TICKER"
    return WHY_CACHE_DIR / f"{safe}-{signature}.json"


def _default_model_fingerprint() -> str:
    parts = [
        os.getenv("BEDROCK_MODEL_ID", "").strip(),
        os.getenv("BEDROCK_LLM_ID_CLAUDE", "").strip(),
        os.getenv("BEDROCK_LLM_ID_OPENAI", "").strip(),
    ]
    filtered = [part for part in parts if part]
    return "|".join(filtered) if filtered else "no-model"


def _atomic_write_json(*, path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    try:
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        path.write_text(data, encoding="utf-8")


def _parse_iso_or_none(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
