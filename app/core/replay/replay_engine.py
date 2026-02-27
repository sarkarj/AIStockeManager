from __future__ import annotations

from typing import Any

from app.core.drl.drl_engine import evaluate_drl
from app.core.replay.artifact_store import compute_policy_hash

try:
    from deepdiff import DeepDiff
except Exception:  # pragma: no cover - exercised only when deepdiff is unavailable
    DeepDiff = None  # type: ignore[assignment]


def replay_artifact(artifact: dict, policy_path: str, now_iso: str) -> dict:
    context_pack = artifact.get("context_pack", {})
    artifact_meta = artifact.get("meta", {})

    metrics = context_pack.get("indicators", {}).get("metrics", {})
    ticker = (
        str(artifact_meta.get("ticker", "")).strip().upper()
        or str(context_pack.get("meta", {}).get("ticker", "")).strip().upper()
    )
    as_of = (
        str(context_pack.get("indicators", {}).get("as_of", "")).strip()
        or str(context_pack.get("prices", {}).get("as_of", "")).strip()
        or str(artifact_meta.get("now_iso", "")).strip()
        or now_iso
    )

    drl_inputs = {
        "ticker": ticker,
        "as_of": as_of,
        **metrics,
    }

    expected_source = artifact.get("drl_result") or context_pack.get("drl", {}).get("result", {})

    actual_full = evaluate_drl(policy_path=policy_path, inputs=drl_inputs, now_iso=now_iso)

    subset_keys = ["action_final", "confidence_cap", "regime_1D", "regime_1W", "gates_triggered", "conflicts"]
    expected = {k: expected_source.get(k) for k in subset_keys}
    actual = {k: actual_full.get(k) for k in subset_keys}

    diff = _compute_diff(expected, actual)
    current_policy_hash = compute_policy_hash(policy_path)
    artifact_policy_hash = str(artifact_meta.get("policy_hash", ""))
    policy_mismatch = bool(artifact_policy_hash and artifact_policy_hash != current_policy_hash)

    ok = diff is None and not policy_mismatch

    result: dict[str, Any] = {
        "ok": ok,
        "expected": expected,
        "actual": actual,
        "diff": diff,
        "policy_mismatch": policy_mismatch,
        "artifact_policy_hash": artifact_policy_hash,
        "current_policy_hash": current_policy_hash,
    }

    if policy_mismatch:
        result["warning"] = "POLICY_HASH_MISMATCH"

    return result


def _compute_diff(expected: dict, actual: dict) -> dict | None:
    if DeepDiff is not None:
        dd = DeepDiff(expected, actual, ignore_order=True)
        if not dd:
            return None
        if hasattr(dd, "to_dict"):
            return dd.to_dict()
        return dict(dd)

    if expected == actual:
        return None

    changes = {}
    all_keys = sorted(set(expected.keys()).union(actual.keys()))
    for key in all_keys:
        if expected.get(key) != actual.get(key):
            changes[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    return {"values_changed": changes}
