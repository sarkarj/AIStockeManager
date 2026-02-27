#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.marketdata.prewarm import (
    acquire_worker_lock,
    cadence_minutes_for_bucket,
    load_prewarm_config,
    load_prewarm_status,
    mark_worker_heartbeat,
    next_due_at,
    pop_prewarm_requests,
    prune_cache,
    release_worker_lock,
    resolve_schedule_bucket,
    run_revalidate_job,
    run_why_refresh_job,
    run_scheduled_refresh,
    save_prewarm_status,
    scheduled_run_due,
)
from app.core.marketdata.query_graph import MarketQueryService
from app.core.orchestration.time_utils import now_iso

_ET = ZoneInfo("America/New_York")


def _build_query_service() -> MarketQueryService:
    return MarketQueryService(
        cache_dir=".cache/charts",
        context_loader=None,
        short_interval="1h",
        short_lookback_days=60,
        long_interval="1h",
        long_lookback_days=60,
        cache_only=False,
    )


def _handle_manual_requests(query: MarketQueryService, config: dict) -> bool:
    handled = False
    requests = pop_prewarm_requests(max_items=10)
    for request in requests:
        tickers = {str(x).strip().upper() for x in (request.get("tickers") or []) if str(x).strip()}
        ranges = tuple(str(x).strip().upper() for x in (request.get("range_keys") or []) if str(x).strip())
        if not ranges:
            ranges = ("1D", "1W")
        reason = str(request.get("reason", "manual")).strip().lower() or "manual"
        if reason == "why_refresh":
            metadata = request.get("metadata", {}) if isinstance(request.get("metadata"), dict) else {}
            expected_signature = str(metadata.get("why_signature", "")).strip().lower() or None
            ticker = next(iter(sorted(tickers)), "")
            result = run_why_refresh_job(
                ticker=ticker,
                range_key=ranges[0] if ranges else "1D",
                expected_signature=expected_signature,
            )
        else:
            result = run_revalidate_job(
                query=query,
                scope=str(request.get("scope", "manual")),
                tickers=tickers,
                range_keys=ranges,
                reason=reason,
            )
        _finalize_status(config=config, result=result, scheduled=False)
        handled = True
    return handled


def _handle_scheduled_refresh(query: MarketQueryService, config: dict) -> bool:
    status = load_prewarm_status()
    due, bucket, now_local = scheduled_run_due(config=config, status=status, now_et=datetime.now(_ET))
    if not due:
        mark_worker_heartbeat(pid=os.getpid(), note=f"idle:{bucket}")
        return False

    result = run_scheduled_refresh(query=query, config=config)
    result["reason"] = f"scheduled:{bucket}"
    result["bucket"] = bucket
    _finalize_status(config=config, result=result, scheduled=True)
    return True


def _finalize_status(*, config: dict, result: dict, scheduled: bool) -> None:
    status = load_prewarm_status()
    status.update(
        {
            "worker_pid": int(os.getpid()),
            "worker_note": "completed",
            "worker_heartbeat_at": now_iso(),
            "last_scope": result.get("scope"),
            "last_reason": result.get("reason"),
            "last_started_at": result.get("started_at"),
            "last_completed_at": result.get("completed_at"),
            "last_tickers": list(result.get("tickers", [])),
            "last_ranges": list(result.get("ranges", [])),
            "attempted": int(result.get("attempted", 0)),
            "live": int(result.get("live", 0)),
            "cache": int(result.get("cache", 0)),
            "none": int(result.get("none", 0)),
            "errors": int(result.get("errors", 0)),
        }
    )
    if scheduled:
        status["last_scheduled_at"] = result.get("completed_at") or now_iso()
    cleanup = prune_cache(
        max_age_hours=int(config.get("cache_max_age_hours", 48)),
        max_budget_mb=int(config.get("cache_budget_mb", 250)),
    )
    status["cache_prune"] = cleanup
    status["next_due_at"] = next_due_at(config=config, status=status)
    save_prewarm_status(status)


def main() -> None:
    while not acquire_worker_lock():
        print("prewarm-worker: lock exists, retrying in 5s")
        time.sleep(5)

    print("prewarm-worker: started")
    try:
        while True:
            config = load_prewarm_config()
            bucket = resolve_schedule_bucket(datetime.now(_ET))
            cadence = cadence_minutes_for_bucket(config, bucket)
            mark_worker_heartbeat(pid=os.getpid(), note=f"{bucket}:{cadence}m")

            if not bool(config.get("enabled", True)):
                time.sleep(15)
                continue

            query = _build_query_service()
            manual_done = _handle_manual_requests(query=query, config=config)
            if manual_done:
                continue

            _handle_scheduled_refresh(query=query, config=config)
            time.sleep(15)
    finally:
        release_worker_lock()
        print("prewarm-worker: stopped")


if __name__ == "__main__":
    main()
