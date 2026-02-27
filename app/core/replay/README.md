# Replay Artifacts (Phase 6)

Replay artifacts capture deterministic DRL context for auditability and regression checks.

## What is stored

Each artifact stores:
- `meta`: ticker, timestamps, policy id/version/hash, app version, notes
- `context_pack`: full context pack JSON snapshot
- `drl_result`: DRL output at snapshot time
- `drl_trace`: DRL decision trace
- `hub_card`: optional hub card snapshot

## Storage layout

Artifacts are written to:

```text
.cache/replay/{TICKER}/{YYYYMMDD-HHMMSS}-{policy_hash}.json
```

Only the latest N artifacts per ticker are retained (currently 30).

## Replay behavior

Replay re-runs `evaluate_drl` using the saved indicator inputs and compares key outputs:
- `action_final`
- `confidence_cap`
- `regime_1D`, `regime_1W`
- `gates_triggered`
- `conflicts`

If values differ, replay returns a structured diff.

## Policy hash meaning

`policy_hash` is `sha256(policy_file_bytes)[:16]`.
It guards against silent policy drift.
If hash differs during replay, replay flags a policy mismatch even if outputs match.

## Scripts

- Save a batch of artifacts: `python -m scripts.snapshot_batch`
- Replay-check artifacts: `python -m scripts.replay_check --ticker AAPL`
