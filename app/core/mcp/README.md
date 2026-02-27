# MCP Server (Phase 3)

This module exposes a read-only MCP server over stdio for stock context data.

## Run

```bash
python -m app.core.mcp.server
```

Wrapper script:

```bash
./scripts/run_mcp_server.sh
```

## Tools

- `stock.get_prices`
  - Params schema: `{ ticker: string (required), interval: string=1h, lookback_days: integer [1..365]=60 }`
- `stock.get_news`
  - Params schema: `{ ticker: string (required), lookback_hours: integer [1..168]=48, max_items: integer [1..30]=12 }`
- `stock.get_macro_snapshot`
  - Params schema: `{ lookback_days: integer [1..30]=1 }`
- `stock.get_events`
  - Params schema: `{ ticker: string (required), max_items: integer [0..20]=10 }`

All tools return normalized JSON objects with `as_of` timestamps and a `source` block.

## Caching

- Disk cache: `DiskTTLCache` under `.cache/`
- Default TTL: `3600` seconds (1 hour)
- Deterministic keys:
  - `prices:{ticker}:{interval}:{lookback_days}`
  - `news:{ticker}:{lookback_hours}:{max_items}`
  - `macro:{lookback_days}`
  - `events:{ticker}:{max_items}`

## Stable IDs (for citations)

- News item ID: `sha1(url)[:16]`
- Event item ID: `sha1(f"{ticker}:{type}:{date}:{title}")[:16]`

These ID rules are deterministic and consistent across runs.
