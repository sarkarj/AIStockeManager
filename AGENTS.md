# AGENTS.md — Instructions for Coding Agents (Codex)

Codex: read this file fully before making changes. This repository is a personal AI stock manager for analysis/monitoring only (**NO trading execution**). The product must be deterministic, auditable, and resilient.

---

## Architecture & Quality Gates (Non-Negotiable)

This repo is governed by **ARCH_ACCEPTANCE.md**. Any change must preserve its principles.
PR Checklist: ✅ make green ✅ DRL integrity PASS ✅ UI contract tests PASS ✅ No DB / no LLM decisioning ✅ No new dependencies without approval

### Must-pass gates (Definition of “Green”)

Before considering work complete, you MUST ensure:

* `make green` passes (runs `pytest -q` and `python3 scripts/verify_drl_integrity.py`)
* No DRL semantic drift (fixtures + invariants + semantic lock must remain PASS)
* UI action/confidence must remain sourced ONLY from `context_pack["drl"]["result"]` via ViewModels

### Forbidden changes (unless explicitly requested)

* Do NOT make LLM decide action/confidence (Hub Card is non-decider)
* Do NOT relax grounding (citations required, forbidden hedge terms enforced)
* Do NOT introduce randomness into DRL
* Do NOT add a database (portfolio.json + disk cache + replay artifacts only)

### Required workflow for risky changes

If editing DRL policy/engine or hub schema:

1. Update/extend fixtures and invariants as needed
2. Run: `make green`
3. Run: `python3 scripts/replay_check.py --ticker <TICKER> --latest` (recommended)

If editing UI:

* Keep all action/confidence display derived from ViewModels.
* `tests/test_ui_contract.py` must stay green.

---

## Non-negotiable product rules

1. Decision is deterministic (DRL).

   * DRL returns `action_final`, `confidence_cap`, `conflicts[]`, `watch_conditions[]`, and a full `decision_trace`.
   * LLM is NOT allowed to decide action. LLM only explains DRL results later.

2. No hallucinated narratives (later LLM layer must cite evidence IDs; degrade otherwise).

3. No AI randomness: same inputs => same DRL outputs.

4. No database. Portfolio stored in `portfolio.json` only.

5. Professional failure modes: no raw stack traces in UI; use DEGRADE/STALE states.

---

## Phase plan

PHASE 1: DRL engine + fixtures + tests ONLY.
Exit criteria: pytest passes; all 12 fixtures pass deterministically.

---

## Setup

* `python -m venv .venv && source .venv/bin/activate`
* `pip install -e .`
* `pytest -q`

---

## Coding conventions

* DRL must have zero network calls.
* DRL evaluation order is strict (see main prompt).
* If fixture fails, fix engine to match policy+fixtures; do NOT “reinterpret” them.

---

## Forbidden

* adding paid data deps
* introducing a database
* letting LLM pick BUY/SELL/HOLD
