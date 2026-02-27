# AIStockManager

Deterministic personal stock analysis platform with cache-first UI, background data prewarm, and grounded AI narrative.

## What this system is

- A personal analytics app for monitoring/analysis.
- DRL (deterministic reconciliation layer) is the only decision-maker.
- LLM (Hub / The Why) explains DRL output and never overrides action/confidence.

## What this system is not

- No brokerage integration.
- No order execution.
- No database backend.

---

## Architecture at a glance

```text
                 +---------------------------+
                 | Yahoo Finance / Bedrock   |
                 +------------+--------------+
                              ^
                              | live fetch (background/API revalidate)
                              |
+------------------+   +------+--------+   +-------------------------+
| Streamlit UI     |-->| Query Graph   |-->| Chart/Quote Fetchers    |
| Pulse/Brain/     |   | service       |   | retry + TTL + disk cache|
| Horizon          |   +------+--------+   +-----------+-------------+
+---------+--------+          |                        |
          |                   v                        v
          |          +--------+---------+      .cache/charts
          |          | Context Pack     |      .cache/why
          |          | Indicators + DRL |      .cache/prewarm
          |          +--------+---------+
          |                   |
          |                   v
          |          +--------+---------+
          |          | Hub generator    |
          +--------->| (optional LLM)   |
                     +------------------+
```

---

## Core principles (enforced)

- DRL-only decisioning: UI action/confidence must come from `context_pack["drl"]["result"]`.
- Deterministic output: same inputs => same DRL outputs.
- Grounded Hub: schema-valid, citation-aware, forbidden-term checks.
- Professional degrade path: explicit stale/tool-down states.
- No DB: JSON state + disk caches only.

Reference: `ARCH_ACCEPTANCE.md`.

---

## Runtime topology

`docker-compose.yml` starts 3 services:

1. `app` (`aistockmanager`)
- Streamlit UI
- Port: `127.0.0.1:8601 -> 8501`
- `MARKETDATA_CACHE_ONLY=1` (UI is cache-first)

2. `graph-api` (`aistockmanager-graph-api`)
- Short/long query endpoints
- Port: `127.0.0.1:8602 -> 8701`
- `MARKETDATA_CACHE_ONLY=0`

3. `prewarm` (`aistockmanager-prewarm`)
- Background worker
- Revalidate fetch + cache hygiene

Caddy routes:
- `https://aistock.localhost`
- `https://api.aistock.localhost`

---

## Cold-start behavior

Because the UI container runs cache-only mode, first launch can show degraded/empty cards if cache is not populated yet.

Data becomes available when:
- prewarm scheduled cycle runs, or
- user clicks **Refresh Data** (queued job), then worker completes.

---

## Query contracts

`app/core/query/contracts.py`

### Short query (`run_short_query`)
Used by Pulse + Brain stage-1 hydration.
Returns:
- ticker
- quote
- series_1d
- series_1w
- drl_result
- context_pack
- as_of

### Long query (`run_long_query`)
Used by Brain full render.
Returns:
- ticker
- range_key
- quote
- selected_series
- drl_result
- context_pack
- why_signature
- as_of

---

## Chart and quote semantics

### Chart range mapping
`app/core/marketdata/chart_fetcher.py`

| Range | period | interval | prepost | TTL |
|---|---|---|---|---|
| 1D | 1d | 5m | true | 300s |
| 1W | 5d | 30m | true | 3600s |
| 1M | 1mo | 1h | true | 3600s |
| 3M | 3mo | 1d | false | 3600s |
| YTD | ytd | 1d | false | 3600s |
| 1Y | 1y | 1d | false | 3600s |

Behavior:
- fresh cache preferred
- live retry/backoff when needed
- stale-cache fallback when live fails
- cooldown to avoid repeated provider hammering
- no synthetic fabricated series in fetcher

### Canonical quote snapshot
`app/core/marketdata/quotes.py`

Fields include: prev_close, close, last_regular, after_hours, latest derivation, session state, quality flags.

Derived:
- `latest_price` from session-aware logic
- `today_change` from regular close vs prev close
- `after_hours_change` from AH vs regular close

### Price sanity
`app/core/marketdata/price_sanity.py`
- flags like `MISSING_BARS`, `PRICE_MISMATCH`
- display/diagnostic safety layer only (does not change DRL math)

---

## UI modules

### The Pulse
`app/ui/components/pulse.py`
- holdings cards
- quote + sparkline + DRL pill
- add/update/delete holdings
- full-card click opens Brain ticker

### The Brain
`app/ui/components/brain.py`
- progressive hydration (stage-1 then stage-2)
- range charts
- Why rendering (Hub or deterministic fallback)
- Evidence/Diagnostics expanders + export buttons

### The Horizon
`app/ui/components/horizon.py`
- scans S&P100 universe
- ranks full set, then takes top-10
- supports metric mode:
  - Today + After-hours
  - Regular Session
- fetches 1D sparklines only after top-10 selection

Ranking formulas:
- `regular_pct = (last_regular - prev_close) / prev_close * 100`
- `today_after_pct = (latest - prev_close) / prev_close * 100`

Sort order:
1. descending `abs(primary_pct)`
2. descending `primary_pct`
3. ascending ticker

---

## Refresh model

### Manual refresh scopes (UI)

- `Visible`: Pulse holdings + selected Brain ticker, ranges `1D,1W`
- `Brain Only`: selected Brain ticker, ranges `1D,1W,1M,3M,YTD,1Y`
- `Horizon Only`: S&P100 universe, range `1W` (ranking basis)
- `Queue All`: broad queued refresh, ranges `1D,1W,1M,3M,YTD,1Y`

All are queue-based. UI does not run heavy live fetch in request thread.

### Worker schedule (ET)
Configurable in UI Configuration panel.
Defaults:
- market hours: 30m
- after-hours: 60m
- off-hours: 12h
- weekends/holidays: 12h

---

## Why/Hub lifecycle

### Signature cache
`app/core/context_pack/why_cache.py`
- deterministic signature from ticker/range/DRL/indicators/quote/model fingerprint
- cache files under `.cache/why`

### Brain Why flow
- check exact signature cache
- if miss: short sync attempt
- if still unresolved: queue background `why_refresh`
- optional latest-cache fallback depending on availability

Diagnostics fields:
- `why_signature_requested`
- `why_signature_loaded`
- `why_source`
- `why_sync_status`
- token counters when available

Status interpretation:
- `why_source=cache_hit`: exact cached artifact loaded
- `why_source=live_sync`: sync generation succeeded
- `why_source=cache_fallback`: latest cached artifact reused
- `why_source=queued/pending`: background refresh queued
- `why_sync_status=timeout/error`: sync path failed; worker path expected
- `why_sync_status=llm_not_configured`: Bedrock config/credentials missing

---

## Graph API

`app/api/graph_api.py`

Endpoints:
- `GET /api/health`
- `POST /api/query/short`
- `POST /api/query/long`

Auth:
- `GRAPH_API_KEY` required by default
- supports `x-api-key` or `Authorization: Bearer ...`
- optional localhost bypass: `GRAPH_API_ALLOW_UNAUTH_LOCALHOST=1`

Revalidate throttle:
- `GRAPH_API_REVALIDATE_COOLDOWN_SECONDS` (default 15)

Error contract:

| HTTP | code | Meaning |
|---|---|---|
| 400 | INVALID_JSON | invalid JSON body |
| 400 | INVALID_PAYLOAD | body is not an object |
| 400 | PAYLOAD_TOO_LARGE | too many fields |
| 401 | UNAUTHORIZED | missing/invalid key |
| 422 | INVALID_TICKER | ticker fails validation |
| 429 | REVALIDATE_THROTTLED | cooldown active |
| 503 | AUTH_NOT_CONFIGURED | key not configured |

---

## Security and sensitive-data posture

Implemented:
- non-root app user in Docker image
- loopback-only host port binding for app/api
- Caddy internal TLS local hostnames
- API key guardrails for Graph API
- input validation for ticker fields
- atomic JSON write with bind-mount fallback
- no DB service exposed

Sensitive-data controls:
- `.env` ignored
- `.cache/` ignored
- `app/core/portfolio/portfolio.json` ignored
- `app/core/portfolio/portfolio.json.tmp` ignored

---

## Build and validation

### Local
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
make green
```

### Docker
```bash
docker compose up --build -d app graph-api prewarm
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Health checks
```bash
curl -fsS http://localhost:8601/_stcore/health
curl -k -fsS https://api.aistock.localhost/api/health
```

### Quality gates
```bash
make green
pytest -q
python3 scripts/verify_drl_integrity.py
```

Optional:
```bash
make llm-smoke
make hub-verify
```

---

## Operations runbook

### Prewarm lock recovery
If queue is stalled and worker lock appears stale:

```bash
rm -f .cache/prewarm/worker.lock .cache/prewarm/queue.lock
docker compose restart prewarm
```

Then verify:
- `.cache/prewarm/status.json` heartbeat updates
- queue depth declines

### Common issue: no chart data
- check worker status in Diagnostics
- queue a scope refresh
- verify graph-api/prewarm has outbound network access

### Common issue: Why fallback persists
- inspect `why_source` / `why_sync_status`
- run `make llm-smoke`
- verify Bedrock env + credentials are present in containers

---

## File map

- UI entry: `app/ui/streamlit_app.py`
- Pulse: `app/ui/components/pulse.py`
- Brain: `app/ui/components/brain.py`
- Horizon: `app/ui/components/horizon.py`
- Query graph: `app/core/marketdata/query_graph.py`
- Query contracts: `app/core/query/contracts.py`
- Chart fetcher: `app/core/marketdata/chart_fetcher.py`
- Quote snapshot: `app/core/marketdata/quotes.py`
- Prewarm scheduler/queue: `app/core/marketdata/prewarm.py`
- Why cache: `app/core/context_pack/why_cache.py`
- Hub generator: `app/core/context_pack/hub_generator.py`
- Graph API: `app/api/graph_api.py`
- DRL integrity: `scripts/verify_drl_integrity.py`

---

## Completeness checklist

- [x] Deterministic DRL-only action/confidence contract
- [x] Pulse/Brain/Horizon backed by shared query graph
- [x] Explicit range mapping + cache strategy
- [x] Queue-based partial refresh model
- [x] Why signature cache + sync/async refresh path
- [x] Hub integrity verifier scripts
- [x] S&P100 full-universe Horizon ranking
- [x] Graph API with auth + throttle
- [x] Security hardening in Docker and ignore rules
- [x] `make green` + DRL integrity gate
