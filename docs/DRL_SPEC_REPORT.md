# DRL Integrity Spec Report

## Scope
This report reverse-audits the deterministic DRL implementation against:
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/policies/drl_policy.yaml`
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py`
- `/Users/jagannathsarkar/Documents/AIStockManager/app/data/fixtures/drl_fixtures.yaml`

Result: **PASS**. No policy-semantic deviations were found.

## A) Deterministic Execution Order (Exact)
The implementation follows the required order exactly in `evaluate_drl`.

1. `validate required inputs`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:261-267`

2. `compute zones`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:274-281`
- Band selection helper: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:74-85`

3. `compute derived features`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:283-288`
- Expression evaluator: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:235-241`

4. `compute regimes (1D/1W) via weighted voting`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:290-339`

5. `compute score components + multipliers + penalties`
- Components: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:341-354`
- Multipliers: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:355-362`
- Penalties: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:363-370`
- Aggregate score: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:371-374`

6. `map score_final to base action via score bands`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:376-385`
- Band resolver: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:244-248`

7. `apply governor rules (weekly)`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:387-398`

8. `apply gates in gates.evaluation_order (including oversold exception)`
- Gate loop: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:399-437`
- Exception handling: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:421-427`

9. `enforce disallowed actions deterministically (fallback to WAIT)`
- Enforce disallowed: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:438-440`
- Fallback ordering helper: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:251-255`

10. `produce decision_trace`
- Build trace: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:457-484`
- Required field enforcement: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:486-489`
- Result object return: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:491-500`

## B) Policy-to-Code Mapping Table
| YAML Path | Code Location | Evaluation | Deviation |
|---|---|---|---|
| `policy.id`, `policy.version` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:31-34`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:459-460` | Validated required; written into trace | NONE |
| `policy.default_profile` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:36-39`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:269-270` | Validated exists in profiles; used to select action profile | NONE |
| `policy.description` | N/A (metadata only) | Not decision-bearing | NONE |
| `inputs.required_metrics[*]` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:47-50`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:262-267` | Required-field validation | NONE |
| `inputs.required_flags[*]` | Same as above | Required-field validation | NONE |
| `normalization.zones.<metric>.ranges[*].{min,max,name}` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:274-281`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:66-78` | Inclusive lower, exclusive upper band match | NONE |
| `normalization.derived_features[*].expression` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:283-288`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:235-241` | Evaluated with deterministic expression parser | NONE |
| `normalization.derived_features[*].name` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:287-288` | Added into evaluation context | NONE |
| `normalization.derived_features[*].output_type`, `enum_values` | N/A (metadata only) | Not used in decision math | NONE |
| `states.regime.timeframes` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:294` | Iterated in order (`1D`, `1W`) | NONE |
| `states.regime.components[*].feature` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:299-301` | `${tf}` expansion to concrete feature name | NONE |
| `states.regime.components[*].bullish_when`, `bearish_when` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:305-306` | Expression evaluation per component | NONE |
| `states.regime.components[*].weight` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:307`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:309-312` | Weighted vote accumulation | NONE |
| `states.regime.voting.{bullish_min_votes,bearish_min_votes,tie_state}` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:325-330` | Thresholded regime state assignment | NONE |
| `scoring.components[*].contributions[*].when` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:348-351` | Condition-gated score contribution | NONE |
| `scoring.components[*].contributions[*].score` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:350` | Numeric additive contribution | NONE |
| `scoring.multipliers[*].metric` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:357-361` | Band-selected multiplier source metric | NONE |
| `scoring.multipliers[*].bands[*].{min,max,multiplier}` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:359`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:81-85` | Range lookup and product accumulation | NONE |
| `scoring.penalties[*].metric` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:365-369` | Band-selected penalty source metric | NONE |
| `scoring.penalties[*].bands[*].{min,max,penalty}` | Same as above + `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:81-85` | Range lookup and additive penalty accumulation | NONE |
| `actions.profiles.swing.score_bands[*].{min,max,action,base_confidence}` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:377-385` | Band selection -> base action and base confidence | NONE |
| `actions.profiles.swing.governor_rules[*].when` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:392-393` | Rule activation condition | NONE |
| `actions.profiles.swing.governor_rules[*].disallow_actions` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:395-396` | Adds to disallowed action set | NONE |
| `actions.profiles.swing.governor_rules[*].confidence_cap_delta` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:397` | Added to confidence delta accumulator | NONE |
| `gates.evaluation_order[*]` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:404` | Gate evaluation order preserved exactly | NONE |
| `gates.definitions.<gate>.trigger_when` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:406` | Expression-gated trigger | NONE |
| `gates.definitions.<gate>.effects.force_action` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:431-433` | Overrides current action | NONE |
| `gates.definitions.<gate>.effects.disallow_actions` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:419`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:428-429` | Appended into disallowed action set | NONE |
| `gates.definitions.<gate>.effects.confidence_cap_delta` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:412` | Added in gate order | NONE |
| `gates.definitions.<gate>.effects.add_conflicts` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:414-417` | Ordered unique conflict accumulation | NONE |
| `gates.definitions.G_OVERSOLD_BOUNCE.exceptions.allow_action_when.action` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:423-426` | Removes disallow entry for matching action when exception condition is true | NONE |
| `gates.definitions.G_OVERSOLD_BOUNCE.exceptions.allow_action_when.when` | Same as above | Exception condition eval | NONE |
| `gates.definitions.G_HIGH_VOL` WAIT bias behavior | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:435-436` | Explicit policy bias implementation | NONE |
| `output.decision_trace_required_fields[*]` | `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:486-489` | Runtime contract enforcement | NONE |

**Deviation status:** **NONE**.

## C) Expression Parsing / Evaluation Spec
### Supported grammar/elements
Implemented by `_split_ternary`, `_eval_expr`, `_SafeEvaluator`:
- Ternary: `condition ? expr_true : expr_false`
- Boolean operators: `and`, `or`, `not`
- Comparisons: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`
- Parentheses via Python AST expression parsing
- Literals: strings, numbers, lists, tuples, booleans, null-like constants
- Variables: identifiers from evaluation context (`inputs`, `zones`, `derived`, `regime`, score variables)

### Implementation locations
- Ternary split/parser: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:88-149`
- Expression dispatcher: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:235-241`
- AST evaluator nodes: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:152-226`

Unsupported expression nodes/operators raise deterministic `ValueError` (no silent fallback).

## D) Gate Semantics Proof
Gate order is fixed by policy and enforced exactly by loop order.

### `G_OVERSOLD_BOUNCE`
- Trigger: `rsi_14_zone == 'OVERSOLD' or stoch_k_zone == 'OVERSOLD'`
- Effect:
  - default disallow `REDUCE`
  - confidence delta `-15`
  - add conflict `OVERSOLD_BOUNCE_RISK`
- Exception:
  - allows `REDUCE` only when `regime_1D == 'BEAR' and regime_1W == 'BEAR' and adx_14_zone in ['STRONG','EXTREME']`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:404-427`

### `G_OVERBOUGHT_CHASE`
- Trigger: overbought RSI or stochastic
- Effect:
  - force action `WAIT`
  - confidence delta `-10`
  - conflict `OVERBOUGHT_PULLBACK_RISK`
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:404-417`, `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:431-433`

### `G_TIMEFRAME_CONFLICT`
- Trigger: `regime_1D != regime_1W and regime_1W != 'NEUTRAL'`
- Effect:
  - force `WAIT`
  - confidence delta `-15`
  - conflict `TIMEFRAME_CONFLICT`
- Code: same gate loop as above.

### `G_HIGH_VOL`
- Trigger: `atr_pct_zone in ['HIGH','EXTREME']`
- Effect:
  - confidence delta `-15`
  - conflict `HIGH_VOLATILITY`
  - explicit bias to `WAIT` via code path
- Code: `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:435-436`

### `G_LOW_PARTICIPATION`
- Trigger: `vroc_14_zone == 'FALLING'`
- Effect:
  - confidence delta `-10`
  - conflict `LOW_PARTICIPATION`
- Code: gate loop.

## E) Confidence Computation Proof
1. Base confidence from score band action
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:382-385`

2. Governor confidence deltas (sum)
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:392-398`

3. Gate confidence deltas in evaluation order (sum)
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:404-413`

4. Action may change (force/disallow), then `confidence_base` is reselected by final action class
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:442-443`

5. Stale override
- if stale (>90 min): force `WAIT`, append `STALE_DATA`, cap confidence to `min(wait_base + deltas, 55)`
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:447-453`

6. Final clamp
- integer rounded then clamped to `[0,100]`
- `/Users/jagannathsarkar/Documents/AIStockManager/app/core/drl/drl_engine.py:455`

## F) Determinism Proof Checklist
- Randomness: **None** (`random`, `numpy.random`, UUID entropy, and time-dependent branching are absent in DRL core)
- Network calls in DRL: **None** (pure policy+input evaluation only)
- Same inputs => same outputs: **True by construction**
  - deterministic parsing/evaluation
  - fixed gate order
  - fixed disallowed fallback order (`WAIT`, `ACCUMULATE`, `REDUCE`)
  - no mutable external state reads in DRL path

## Verifier Coverage Summary
The integrity verifier (`/Users/jagannathsarkar/Documents/AIStockManager/scripts/verify_drl_integrity.py`) enforces:
- Fixture lock (12 fixtures)
- Invariants lock (INV1..INV5)
- Policy hash roundtrip lock
- Semantic lock against an independent in-script reference implementation across:
  - zone boundaries: RSI, Stoch, ADX, ATR, VROC
  - all gates (+ oversold exception)
  - both governor rules

