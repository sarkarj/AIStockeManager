from __future__ import annotations

import json

import streamlit as st

from app.core.orchestration.time_utils import now_iso
from app.core.replay.artifact_store import compute_policy_hash, list_artifacts, load_artifact, save_artifact
from app.core.replay.replay_engine import replay_artifact


def render_replay_tools(selected_ticker: str, context_pack: dict, policy_path: str) -> None:
    st.markdown("**Replay Tools**")

    note_key = f"replay_note_{selected_ticker}"
    save_key = f"replay_save_{selected_ticker}"
    select_key = f"replay_select_{selected_ticker}"
    run_key = f"replay_run_{selected_ticker}"
    ignore_key = f"replay_ignore_policy_{selected_ticker}"

    note_text = st.text_input("Snapshot Note (optional)", key=note_key, placeholder="e.g., manual check after policy tune")

    if st.button("💾 Save Snapshot", key=save_key):
        notes = [note_text] if note_text.strip() else []
        path = save_artifact(
            ticker=selected_ticker,
            policy_path=policy_path,
            context_pack=context_pack,
            now_iso=now_iso(),
            notes=notes,
        )
        st.session_state[f"replay_last_saved_path_{selected_ticker}"] = path
        st.success(f"Saved snapshot: {path}")

    artifacts = list_artifacts(selected_ticker)
    if not artifacts:
        st.caption("No snapshots saved for this ticker yet.")
        return

    selected_path = st.selectbox("Replay Snapshot", options=artifacts, key=select_key)
    ignore_policy_mismatch = st.checkbox("Ignore policy hash mismatch", value=False, key=ignore_key)

    if st.button("▶ Replay", key=run_key):
        artifact = load_artifact(selected_path)
        result = replay_artifact(artifact=artifact, policy_path=policy_path, now_iso=now_iso())

        current_policy_hash = compute_policy_hash(policy_path)
        artifact_policy_hash = str(artifact.get("meta", {}).get("policy_hash", ""))
        if artifact_policy_hash and artifact_policy_hash != current_policy_hash:
            st.warning(
                f"Policy hash mismatch: artifact={artifact_policy_hash}, current={current_policy_hash}."
            )

        if result.get("ok"):
            st.success("Replay matched expected DRL output.")
            return

        only_policy_mismatch = bool(result.get("policy_mismatch") and not result.get("diff"))
        if only_policy_mismatch and ignore_policy_mismatch:
            st.success("Replay values matched; policy mismatch ignored by user option.")
            return

        st.error("Replay mismatch detected.")
        st.markdown("**Expected vs Actual**")
        col_a, col_b = st.columns(2)
        col_a.json(result.get("expected", {}), expanded=True)
        col_b.json(result.get("actual", {}), expanded=True)

        diff = result.get("diff")
        if diff:
            st.markdown("**Diff**")
            st.code(json.dumps(diff, indent=2, ensure_ascii=True), language="json")
