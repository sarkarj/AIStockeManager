# Intelligence Hub Card Schema

The Intelligence Hub Card is a structured explanation layer for DRL decisions. It does not decide actions; it explains DRL outputs with grounded evidence.

## Schema fields

- `meta`: ticker, generation timestamp, policy metadata, and generation mode (`FULL`, `TECHNICAL_ONLY`, `DEGRADED`).
- `summary`: immutable DRL decision summary (`action_final`, `confidence_cap`) plus one-line explanation.
- `drivers`: key supportive bullets with mandatory citations.
- `conflicts`: risk/conflict bullets with mandatory citations.
- `watch`: monitoring bullets with mandatory citations.
- `evidence.used_ids`: evidence IDs used by bullets.

All objects are strict (`additionalProperties: false`).

## Grounding rules

- Every bullet in `drivers`, `conflicts`, and `watch` must include at least one citation.
- Citation IDs must exist in the context-pack evidence catalog.
- Citation ID formats:
  - `indicator:<name>`
  - `news:<id>`
  - `macro:<id>`
- `summary.action_final` and `summary.confidence_cap` must match DRL exactly.

## Forbidden terms

The output must not contain hedge/speculative terms:
- `could`
- `might`
- `potentially`
- `possibly`
- `speculate`

## Reliability flow

1. Generate structured JSON from Bedrock.
2. Validate schema and grounding constraints.
3. Retry once with stricter prompt if validation fails.
4. If it still fails, return deterministic fallback template (technical-only or degraded).
