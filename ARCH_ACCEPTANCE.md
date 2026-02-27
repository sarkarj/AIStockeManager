# Architecture Acceptance Criteria (ARCH_ACCEPTANCE.md)

This repo implements a **professional-grade personal AI Stock Manager** with **deterministic decisioning** and **grounded explanations**.
The goal is **trust**: no hallucinations, no randomness, and no silent logic drift.

This document defines the **non-negotiable architecture principles** and the **acceptance checks** that must pass for any change to be considered “correct”.

---

## 0) Definitions

* **DRL** = Deterministic Reconciliation Layer. The **only decider** of action + confidence.
* **Hub Card** = LLM-generated explanation. **Non-decider**. Must be grounded and schema-valid.
* **Context Pack** = deterministic bundle of prices + indicators + DRL result + optional hub card + data_quality.

Internal DRL actions:

* `ACCUMULATE`, `WAIT`, `REDUCE`

UI display labels (view-only mapping):

* `ACCUMULATE → BUY`
* `WAIT → HOLD`
* `REDUCE → SELL`

---

## 1) Non-Negotiable Principles

### P1 — DRL is the only decider

**Must always be true:**

* The UI action label and confidence shown to the user are derived **only** from:

  * `context_pack["drl"]["result"]["action_final"]`
  * `context_pack["drl"]["result"]["confidence_cap"]`
* No LLM output (Hub Card or otherwise) can alter action/confidence.
* No UI module recomputes action/confidence.

**Enforcement:**

* `tests/test_ui_contract.py`
* `scripts/verify_drl_integrity.py`
* DRL fixtures + invariants

---

### P2 — No hallucinated narratives (groundedness)

**Must always be true:**

* Hub Card output must be **JSON schema valid**.
* Every explanation bullet must include **citations**.
* Citations must reference only evidence IDs that exist in the provided context pack:

  * `indicator:<name>`
  * `news:<id>`
  * `macro:<id>`
* Forbidden hedge terms are not allowed in Hub Card text (e.g., “could”, “might”, “possibly”, etc.).
* If news/macro is missing or tools are down, Hub Card must degrade to **TECHNICAL_ONLY** or **DEGRADED** rather than fabricate.

**Enforcement:**

* `app/core/llm/hub_card_schema.py` + generator validation
* Hub Card tests (schema + forbidden terms + citations validation)

---

### P3 — Determinism (no randomness)

**Must always be true:**

* DRL is deterministic: same inputs + same policy ⇒ same outputs.
* LLM calls (if enabled) are configured for deterministic behavior:

  * temperature defaults to `0.0`
  * schema validation + retry-once + deterministic fallback

**Enforcement:**

* `scripts/verify_drl_integrity.py` (semantic lock)
* Replay tools (`.cache/replay`) + replay check
* DRL fixtures + invariants

---

### P4 — Reliability UX (degradation + trust)

**Must always be true:**

* Staleness/tool-down states are **explicit**, not hidden:

  * `meta.data_quality.*` must represent freshness and notes.
* When stale or tool-down:

  * DRL must degrade appropriately (e.g., WAIT + confidence cap when stale per policy).
  * UI must show trust badges without changing DRL output.
* UI must never show raw stack traces.

**Enforcement:**

* Freshness + data_quality fields in Context Pack
* Trust badge computation
* UI contract tests

---

### P5 — No database

**Must always be true:**

* Portfolio state is stored in JSON only (atomic writes).
* Cache is disk-based TTL cache only.
* Replay artifacts are stored under `.cache/replay/`.
* No SQL/ORM/DB services are introduced.

**Enforcement:**

* Code review gate + dependency audit
* Repository structure expectations (portfolio.json, .cache/*)

---

## 2) Source of Truth Data Flow

**UI does not decide. UI displays.**

1. UI selects ticker
2. Context Pack builder fetches prices (cached) + computes indicators
3. DRL evaluates → action/confidence + trace
4. Hub Card (optional) explains the DRL result (cannot override it)
5. UI ViewModels map DRL action to BUY/HOLD/SELL for display only

---

## 3) “Green” Definition (Must Pass Before Merge)

Run:

* `make green`

This must run:

* `pytest -q`
* `python3 scripts/verify_drl_integrity.py`

**If either fails, the change is not acceptable.**

Build command intent:

* `make green` = offline deterministic quality gate (no live cloud dependency)
* `make llm-smoke` = optional live Bedrock connectivity check

Optional but recommended before releases:

* `python3 scripts/replay_check.py --ticker <TICKER> --latest`

---

## 4) DRL Integrity Requirements

### DRL execution order is fixed

DRL must follow the defined sequence:

1. validate inputs
2. zones
3. derived features
4. regimes (1D/1W voting)
5. scoring + multipliers/penalties
6. score bands → base action
7. governor rules
8. gates (ordered, with oversold exception)
9. enforce disallowed actions → fallback WAIT
10. decision_trace

**Proof artifact:**

* `docs/DRL_SPEC_REPORT.md`

### DRL semantic lock must remain intact

The verifier must include:

* fixture lock (12 fixtures)
* invariants lock (INV1–INV5)
* policy hash lock
* semantic lock reference DRL with boundary cases + gate cases + governor cases

**Proof artifact:**

* `scripts/verify_drl_integrity.py`

---

## 5) UI Contract Requirements

### ViewModels are mandatory

UI components must consume:

* `app/ui/viewmodels/brain_vm.py`
* `app/ui/viewmodels/pulse_vm.py`
  (and if present, horizon VM or shared mapping utilities)

UI components must not directly decide action/confidence. They may format and map display labels only.

**Enforcement:**

* `tests/test_ui_contract.py`

### Display mapping (view-only)

UI must display:

* BUY/HOLD/SELL, derived only from DRL action_final mapping:

  * ACCUMULATE→BUY, WAIT→HOLD, REDUCE→SELL
* Brain header must show a tiny internal label under the pill:

  * `DRL: ACCUMULATE|WAIT|REDUCE`

---

## 6) Change Safety Rules

### Allowed changes without adding new tests

* UI layout/styling changes (CSS, cards, charts) that do not change ViewModel source-of-truth.
* Performance improvements that do not change DRL outputs.

### Changes that REQUIRE updating tests and verifier expectations

* Any DRL policy change (thresholds, gates, bands, governor rules)
* Any change to DRL computation code
* Any change to hub schema or grounding rules

### Forbidden changes (unless explicitly approved)

* Making the LLM decide action/confidence
* Removing citations requirement
* Introducing randomness in DRL
* Adding a database

---

## 7) Operational Notes

* Use `python3` for scripts (macOS portability).
* `.cache/` is non-critical and can be deleted to reset cache/artifacts.
* Portfolio is intended to remain small (personal use).

---

## 8) Quick Acceptance Checklist (Human Review)

Before calling a release “done”, confirm:

* [ ] `make green` passes
* [ ] UI shows BUY/HOLD/SELL but Brain shows tiny `DRL: ...` line
* [ ] When hub card is missing, UI shows TECHNICAL_ONLY/DEGRADED without crashing
* [ ] Evidence panel shows indicators/news/macro with stable IDs
* [ ] Stale/tool-down states show badges; action/conf remains DRL-driven
* [ ] No DB introduced; portfolio.json remains the only persistent state besides cache/replay

---
