from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from app.core.orchestration.time_utils import parse_iso
from app.core.replay.artifact_schema import ArtifactMeta, ReplayArtifact

ARTIFACT_DIR = ".cache/replay"
MAX_ARTIFACTS_PER_TICKER = 30


def compute_policy_hash(policy_path: str) -> str:
    data = Path(policy_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def save_artifact(
    ticker: str,
    policy_path: str,
    context_pack: dict,
    now_iso: str,
    notes: list[str] | None = None,
) -> str:
    ticker_norm = ticker.strip().upper()
    policy_hash = compute_policy_hash(policy_path)

    trace = context_pack.get("drl", {}).get("decision_trace", {})
    drl_result = context_pack.get("drl", {}).get("result", {})

    policy_id = str(trace.get("policy_id") or _policy_value(policy_path, "id") or "")
    policy_version = str(trace.get("policy_version") or _policy_value(policy_path, "version") or "")

    merged_notes = list(notes or [])
    merged_notes.extend(_source_notes_from_context(context_pack))

    meta = ArtifactMeta(
        ticker=ticker_norm,
        created_at=now_iso,
        now_iso=now_iso,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_hash=policy_hash,
        app_version="0.1.0",
        notes=merged_notes,
    )

    artifact = ReplayArtifact(
        meta=meta,
        context_pack=context_pack,
        drl_result=drl_result,
        drl_trace=trace,
        hub_card=context_pack.get("hub_card"),
    )

    ticker_dir = Path(ARTIFACT_DIR) / ticker_norm
    ticker_dir.mkdir(parents=True, exist_ok=True)

    dt = parse_iso(now_iso)
    fname = f"{dt.strftime('%Y%m%d-%H%M%S')}-{policy_hash}.json"
    out_path = ticker_dir / fname

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(mode="json", exclude_none=True), f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")

    os.replace(tmp_path, out_path)
    _prune_old_artifacts(ticker_dir=ticker_dir)

    return str(out_path)


def list_artifacts(ticker: str) -> list[str]:
    ticker_norm = ticker.strip().upper()
    ticker_dir = Path(ARTIFACT_DIR) / ticker_norm
    if not ticker_dir.exists():
        return []

    files = [p for p in ticker_dir.glob("*.json") if p.is_file()]
    files_sorted = sorted(files, key=lambda p: p.name, reverse=True)
    return [str(p) for p in files_sorted]


def load_artifact(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _policy_value(policy_path: str, key: str) -> str | None:
    try:
        raw = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    if isinstance(raw, dict):
        policy = raw.get("policy", {})
        if isinstance(policy, dict):
            value = policy.get(key)
            if value is not None:
                return str(value)
    return None


def _source_notes_from_context(context_pack: dict) -> list[str]:
    notes: list[str] = []

    prices_source = context_pack.get("prices", {}).get("source")
    if isinstance(prices_source, dict):
        provider = prices_source.get("provider", "unknown")
        notes.append(f"prices_provider:{provider}")
    else:
        notes.append("prices_provider:unknown")

    for channel in ["news", "macro", "events"]:
        source = context_pack.get(channel, {}).get("source")
        if isinstance(source, dict):
            provider = source.get("provider", "unknown")
            notes.append(f"{channel}_provider:{provider}")

    if context_pack.get("hub_card") is not None:
        notes.append("hub_card:present")
    else:
        notes.append("hub_card:absent")

    return notes


def _prune_old_artifacts(ticker_dir: Path) -> None:
    files = sorted([p for p in ticker_dir.glob("*.json") if p.is_file()], key=lambda p: p.name, reverse=True)
    for stale in files[MAX_ARTIFACTS_PER_TICKER:]:
        try:
            stale.unlink()
        except OSError:
            continue
