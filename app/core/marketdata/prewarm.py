from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date, datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.core.marketdata.query_graph import MarketQueryService
from app.core.orchestration.time_utils import now_iso, parse_iso
from app.core.portfolio.portfolio_store import load_portfolio

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = dt_time(9, 30)
_RTH_CLOSE = dt_time(16, 0)
_AH_CLOSE = dt_time(20, 0)
_NYSE_HOLIDAYS = {
    date(2025, 1, 1),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 4, 18),
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),
    date(2027, 7, 5),
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),
}

PREWARM_DIR = Path(".cache/prewarm")
PREWARM_CONFIG_PATH = PREWARM_DIR / "config.json"
PREWARM_QUEUE_PATH = PREWARM_DIR / "queue.json"
PREWARM_QUEUE_LOCK_PATH = PREWARM_DIR / "queue.lock"
PREWARM_STATUS_PATH = PREWARM_DIR / "status.json"
PREWARM_LOCK_PATH = PREWARM_DIR / "worker.lock"
BRAIN_RANGE_KEYS: tuple[str, ...] = ("1D", "1W", "1M", "3M", "YTD", "1Y")

DEFAULT_PREWARM_CONFIG: dict[str, Any] = {
    "enabled": True,
    "market_minutes": 30,
    "after_hours_minutes": 60,
    "off_hours_minutes": 720,
    "weekend_minutes": 720,
    "cache_max_age_hours": 48,
    "cache_budget_mb": 250,
    "horizon_enabled": True,
    "horizon_universe": "sp100",
    "horizon_range_keys": ["1W"],
    "main_range_keys": list(BRAIN_RANGE_KEYS),
}


def ensure_prewarm_paths() -> None:
    PREWARM_DIR.mkdir(parents=True, exist_ok=True)


def load_prewarm_config() -> dict[str, Any]:
    ensure_prewarm_paths()
    if not PREWARM_CONFIG_PATH.exists():
        return dict(DEFAULT_PREWARM_CONFIG)
    try:
        payload = json.loads(PREWARM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_PREWARM_CONFIG)
    if not isinstance(payload, dict):
        return dict(DEFAULT_PREWARM_CONFIG)
    merged = dict(DEFAULT_PREWARM_CONFIG)
    merged.update(payload)
    merged["market_minutes"] = _safe_int(merged.get("market_minutes"), 30, minimum=5)
    merged["after_hours_minutes"] = _safe_int(merged.get("after_hours_minutes"), 60, minimum=5)
    merged["off_hours_minutes"] = _safe_int(merged.get("off_hours_minutes"), 720, minimum=30)
    merged["weekend_minutes"] = _safe_int(merged.get("weekend_minutes"), 720, minimum=30)
    merged["cache_max_age_hours"] = _safe_int(merged.get("cache_max_age_hours"), 48, minimum=1)
    merged["cache_budget_mb"] = _safe_int(merged.get("cache_budget_mb"), 250, minimum=50)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["horizon_enabled"] = bool(merged.get("horizon_enabled", True))
    merged["main_range_keys"] = _ensure_brain_ranges(
        _normalize_ranges(merged.get("main_range_keys"), fallback=BRAIN_RANGE_KEYS)
    )
    merged["horizon_range_keys"] = _normalize_ranges(merged.get("horizon_range_keys"), fallback=("1W",))
    return merged


def save_prewarm_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(load_prewarm_config())
    merged.update(dict(config or {}))
    normalized = load_prewarm_config_from_payload(merged)
    _atomic_write_json(PREWARM_CONFIG_PATH, normalized)
    return normalized


def load_prewarm_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_PREWARM_CONFIG)
    merged.update(dict(payload or {}))
    merged["market_minutes"] = _safe_int(merged.get("market_minutes"), 30, minimum=5)
    merged["after_hours_minutes"] = _safe_int(merged.get("after_hours_minutes"), 60, minimum=5)
    merged["off_hours_minutes"] = _safe_int(merged.get("off_hours_minutes"), 720, minimum=30)
    merged["weekend_minutes"] = _safe_int(merged.get("weekend_minutes"), 720, minimum=30)
    merged["cache_max_age_hours"] = _safe_int(merged.get("cache_max_age_hours"), 48, minimum=1)
    merged["cache_budget_mb"] = _safe_int(merged.get("cache_budget_mb"), 250, minimum=50)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["horizon_enabled"] = bool(merged.get("horizon_enabled", True))
    merged["main_range_keys"] = _ensure_brain_ranges(
        _normalize_ranges(merged.get("main_range_keys"), fallback=BRAIN_RANGE_KEYS)
    )
    merged["horizon_range_keys"] = _normalize_ranges(merged.get("horizon_range_keys"), fallback=("1W",))
    return merged


def enqueue_prewarm_request(
    *,
    scope: str,
    tickers: set[str],
    range_keys: tuple[str, ...],
    reason: str,
    requested_by: str = "ui",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_prewarm_paths()
    acquired = _acquire_queue_lock()
    queue = _load_queue_payload()
    request = {
        "request_id": str(uuid.uuid4()),
        "queued_at": now_iso(),
        "scope": str(scope or "visible").strip().lower() or "visible",
        "reason": str(reason or "manual").strip().lower() or "manual",
        "requested_by": str(requested_by or "ui").strip().lower() or "ui",
        "tickers": sorted({str(t or "").strip().upper() for t in tickers if str(t or "").strip()}),
        "range_keys": list(_normalize_ranges(range_keys, fallback=("1D", "1W"))),
    }
    if isinstance(metadata, dict) and metadata:
        request["metadata"] = metadata
    queue.append(request)
    try:
        _atomic_write_json(PREWARM_QUEUE_PATH, {"queue": queue})
    finally:
        if acquired:
            _release_queue_lock()
    return request


def pop_prewarm_requests(*, max_items: int = 5) -> list[dict[str, Any]]:
    ensure_prewarm_paths()
    acquired = _acquire_queue_lock()
    queue = _load_queue_payload()
    if not queue:
        if acquired:
            _release_queue_lock()
        return []
    take = max(1, int(max_items))
    selected = queue[:take]
    remaining = queue[take:]
    try:
        _atomic_write_json(PREWARM_QUEUE_PATH, {"queue": remaining})
    finally:
        if acquired:
            _release_queue_lock()
    return selected


def prewarm_queue_depth() -> int:
    ensure_prewarm_paths()
    acquired = _acquire_queue_lock()
    try:
        return len(_load_queue_payload())
    finally:
        if acquired:
            _release_queue_lock()


def cache_hygiene_snapshot(cache_dirs: tuple[Path, ...] | None = None) -> dict[str, Any]:
    dirs = cache_dirs or (Path(".cache/charts"), Path(".cache/quotes"), Path(".cache/why"), Path(".cache/hub_cards"), Path(".cache/context"))
    per_dir: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    total_empty = 0

    for base in dirs:
        path = Path(base)
        file_count = 0
        size_bytes = 0
        empty_files = 0
        if path.exists():
            for item in path.rglob("*.json"):
                if not item.is_file():
                    continue
                file_count += 1
                try:
                    size = int(item.stat().st_size)
                except OSError:
                    size = 0
                size_bytes += max(0, size)
                if size <= 0:
                    empty_files += 1

        total_files += file_count
        total_bytes += size_bytes
        total_empty += empty_files
        per_dir.append(
            {
                "dir": str(path),
                "files": int(file_count),
                "size_mb": round(float(size_bytes) / (1024.0 * 1024.0), 3),
                "empty_files": int(empty_files),
            }
        )

    return {
        "file_count": int(total_files),
        "total_size_mb": round(float(total_bytes) / (1024.0 * 1024.0), 3),
        "empty_files": int(total_empty),
        "dirs": per_dir,
    }


def load_prewarm_status() -> dict[str, Any]:
    ensure_prewarm_paths()
    if not PREWARM_STATUS_PATH.exists():
        return {}
    try:
        payload = json.loads(PREWARM_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_prewarm_status(status: dict[str, Any]) -> None:
    ensure_prewarm_paths()
    _atomic_write_json(PREWARM_STATUS_PATH, dict(status or {}))


def mark_worker_heartbeat(*, pid: int, note: str = "running") -> None:
    status = load_prewarm_status()
    status["worker_pid"] = int(pid)
    status["worker_note"] = str(note)
    status["worker_heartbeat_at"] = now_iso()
    save_prewarm_status(status)


def acquire_worker_lock(*, max_age_seconds: int = 7200) -> bool:
    ensure_prewarm_paths()
    now_ts = time.time()
    if PREWARM_LOCK_PATH.exists():
        try:
            age = now_ts - PREWARM_LOCK_PATH.stat().st_mtime
        except OSError:
            age = 0
        if age > float(max_age_seconds):
            try:
                PREWARM_LOCK_PATH.unlink()
            except OSError:
                return False
        else:
            return False
    try:
        PREWARM_LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except OSError:
        return False


def release_worker_lock() -> None:
    try:
        if PREWARM_LOCK_PATH.exists():
            PREWARM_LOCK_PATH.unlink()
    except OSError:
        return


def resolve_schedule_bucket(now_et: datetime | None = None) -> str:
    local = (now_et or datetime.now(_ET)).astimezone(_ET)
    if local.weekday() >= 5 or local.date() in _NYSE_HOLIDAYS:
        return "weekend"
    t = local.time()
    if _RTH_OPEN <= t < _RTH_CLOSE:
        return "market"
    if _RTH_CLOSE <= t < _AH_CLOSE:
        return "after_hours"
    return "off_hours"


def cadence_minutes_for_bucket(config: dict[str, Any], bucket: str) -> int:
    cfg = load_prewarm_config_from_payload(config)
    lookup = {
        "market": int(cfg["market_minutes"]),
        "after_hours": int(cfg["after_hours_minutes"]),
        "off_hours": int(cfg["off_hours_minutes"]),
        "weekend": int(cfg["weekend_minutes"]),
    }
    return int(lookup.get(str(bucket), int(cfg["off_hours_minutes"])))


def scheduled_run_due(*, config: dict[str, Any], status: dict[str, Any], now_et: datetime | None = None) -> tuple[bool, str, datetime]:
    local = (now_et or datetime.now(_ET)).astimezone(_ET)
    bucket = resolve_schedule_bucket(local)
    interval_minutes = cadence_minutes_for_bucket(config, bucket)
    last_iso = str(status.get("last_scheduled_at", "")).strip()
    if not last_iso:
        return True, bucket, local
    try:
        last_dt = parse_iso(last_iso).astimezone(_ET)
    except Exception:
        return True, bucket, local
    elapsed_minutes = (local - last_dt).total_seconds() / 60.0
    return elapsed_minutes >= float(interval_minutes), bucket, local


def run_revalidate_job(
    *,
    query: MarketQueryService,
    scope: str,
    tickers: set[str],
    range_keys: tuple[str, ...],
    reason: str,
) -> dict[str, Any]:
    started = now_iso()
    stats = query.revalidate_tickers(tickers=set(tickers), range_keys=tuple(range_keys))
    finished = now_iso()
    return {
        "request_id": str(uuid.uuid4()),
        "scope": str(scope),
        "reason": str(reason),
        "started_at": started,
        "completed_at": finished,
        "tickers": sorted(set(tickers)),
        "ranges": list(range_keys),
        "attempted": int(stats.get("attempted", 0)),
        "live": int(stats.get("live", 0)),
        "cache": int(stats.get("cache", 0)),
        "none": int(stats.get("none", 0)),
        "errors": int(stats.get("errors", 0)),
    }


def run_why_refresh_job(
    *,
    ticker: str,
    range_key: str = "1D",
    expected_signature: str | None = None,
) -> dict[str, Any]:
    from app.core.context_pack.why_cache import build_why_signature, save_why_artifact
    from app.core.marketdata.quotes import get_quote_snapshot, quote_snapshot_to_dict
    from app.core.query.context_loader import load_context_pack_for_query

    started = now_iso()
    symbol = str(ticker or "").strip().upper()
    signature = str(expected_signature or "").strip().lower()
    errors: list[str] = []
    saved = False
    status = "skipped"
    cache_path = ""

    if not symbol:
        errors.append("empty_ticker")
    else:
        try:
            context_pack = load_context_pack_for_query(
                ticker=symbol,
                generate_hub_card=True,
                interval="1h",
                lookback_days=60,
            )
            quote = quote_snapshot_to_dict(get_quote_snapshot(ticker=symbol))
            drl_result = context_pack.get("drl", {}).get("result", {})
            indicators = context_pack.get("indicators", {})
            resolved_signature = build_why_signature(
                ticker=symbol,
                drl_result=drl_result if isinstance(drl_result, dict) else {},
                indicators=indicators if isinstance(indicators, dict) else {},
                quote=quote if isinstance(quote, dict) else {},
                range_key=str(range_key or "1D"),
            )
            target_signature = str(signature or "").strip().lower()
            if not target_signature and resolved_signature:
                target_signature = str(resolved_signature).strip().lower()
            hub_card = context_pack.get("hub_card")
            hub_meta = context_pack.get("meta", {}).get("hub", {})
            if isinstance(hub_card, dict):
                cache_path = save_why_artifact(
                    signature=target_signature,
                    ticker=symbol,
                    hub_card=hub_card,
                    hub_meta=hub_meta if isinstance(hub_meta, dict) else {},
                    generated_at=started,
                )
                if (
                    resolved_signature
                    and str(resolved_signature).strip().lower() != target_signature
                ):
                    save_why_artifact(
                        signature=str(resolved_signature).strip().lower(),
                        ticker=symbol,
                        hub_card=hub_card,
                        hub_meta=hub_meta if isinstance(hub_meta, dict) else {},
                        generated_at=started,
                    )
                saved = bool(cache_path)
                status = "saved" if saved else "cache_write_failed"
                if not saved:
                    errors.append("cache_write_failed")
                signature = target_signature
            else:
                status = "hub_missing"
                reason = str(context_pack.get("meta", {}).get("hub", {}).get("reason", "")).strip()
                if reason:
                    errors.append(reason)
        except Exception as exc:  # noqa: PERF203
            status = "error"
            errors.append(str(exc))

    finished = now_iso()
    return {
        "request_id": str(uuid.uuid4()),
        "scope": "brain",
        "reason": "why_refresh",
        "started_at": started,
        "completed_at": finished,
        "tickers": [symbol] if symbol else [],
        "ranges": [str(range_key or "1D").upper()],
        "attempted": 1 if symbol else 0,
        "live": 1 if saved else 0,
        "cache": 0,
        "none": 0 if saved else (1 if symbol else 0),
        "errors": 1 if errors else 0,
        "why_status": status,
        "why_signature": signature,
        "why_cache_path": cache_path,
        "why_error": errors[0] if errors else None,
    }


def run_scheduled_refresh(*, query: MarketQueryService, config: dict[str, Any], selected_ticker: str | None = None) -> dict[str, Any]:
    cfg = load_prewarm_config_from_payload(config)
    holdings = _portfolio_tickers()
    selected = str(selected_ticker or "").strip().upper()
    if selected:
        holdings.add(selected)

    aggregate = {
        "scope": "scheduled",
        "reason": "scheduled",
        "started_at": now_iso(),
        "completed_at": now_iso(),
        "tickers": sorted(holdings),
        "ranges": list(cfg["main_range_keys"]),
        "attempted": 0,
        "live": 0,
        "cache": 0,
        "none": 0,
        "errors": 0,
    }

    if holdings:
        base = query.revalidate_tickers(tickers=holdings, range_keys=tuple(cfg["main_range_keys"]))
        _merge_stats(aggregate, base)

    if bool(cfg.get("horizon_enabled", True)):
        horizon_tickers = _load_sp100_universe()
        if horizon_tickers:
            h_stats = query.revalidate_tickers(tickers=horizon_tickers, range_keys=tuple(cfg["horizon_range_keys"]))
            _merge_stats(aggregate, h_stats)
    aggregate["completed_at"] = now_iso()
    return aggregate


def prune_cache(*, max_age_hours: int, max_budget_mb: int, cache_dirs: tuple[Path, ...] | None = None) -> dict[str, int]:
    dirs = cache_dirs or (Path(".cache/charts"), Path(".cache/quotes"), Path(".cache/hub"), Path(".cache/context"))
    scanned = 0
    removed = 0
    now_dt = datetime.now(_ET)
    max_age_seconds = max(1, int(max_age_hours)) * 3600

    for base in dirs:
        if not base.exists():
            continue
        for item in base.rglob("*.json"):
            if not item.is_file():
                continue
            scanned += 1
            mtime = item.stat().st_mtime

            try:
                size = item.stat().st_size
            except OSError:
                continue
            if size == 0:
                try:
                    item.unlink()
                    removed += 1
                except OSError:
                    pass
                continue

            age_seconds = now_dt.timestamp() - float(mtime)
            if age_seconds > float(max_age_seconds):
                try:
                    item.unlink()
                    removed += 1
                except OSError:
                    pass

    # Size cap prune (oldest first)
    budget_bytes = max(50, int(max_budget_mb)) * 1024 * 1024
    existing: list[tuple[Path, float, int]] = []
    total = 0
    for base in dirs:
        if not base.exists():
            continue
        for item in base.rglob("*.json"):
            if not item.is_file():
                continue
            try:
                stat = item.stat()
            except OSError:
                continue
            existing.append((item, stat.st_mtime, stat.st_size))
            total += int(stat.st_size)
    if total > budget_bytes:
        for path, _, size in sorted(existing, key=lambda v: v[1]):
            if total <= budget_bytes:
                break
            try:
                path.unlink()
                total -= int(size)
                removed += 1
            except OSError:
                continue

    return {"scanned": int(scanned), "removed": int(removed), "bytes": int(total)}


def next_due_at(*, config: dict[str, Any], status: dict[str, Any], now_et: datetime | None = None) -> str:
    local = (now_et or datetime.now(_ET)).astimezone(_ET)
    bucket = resolve_schedule_bucket(local)
    cadence = cadence_minutes_for_bucket(config, bucket)
    last_iso = str(status.get("last_scheduled_at", "")).strip()
    if not last_iso:
        return local.isoformat()
    try:
        last_dt = parse_iso(last_iso).astimezone(_ET)
    except Exception:
        return local.isoformat()
    return (last_dt + timedelta(minutes=max(1, int(cadence)))).isoformat()


def _portfolio_tickers() -> set[str]:
    portfolio = load_portfolio()
    return {
        str(item.ticker).strip().upper()
        for item in list(getattr(portfolio, "holdings", []) or [])
        if str(getattr(item, "ticker", "")).strip()
    }


def _load_sp100_universe() -> set[str]:
    path = Path("app/data/sp100_universe.json")
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(payload, list):
        return {str(item).strip().upper() for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        tickers = payload.get("tickers", [])
        if isinstance(tickers, list):
            return {str(item).strip().upper() for item in tickers if str(item).strip()}
    return set()


def _normalize_ranges(values: Any, fallback: tuple[str, ...]) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        ranges = [str(v).strip().upper() for v in values if str(v).strip()]
    elif isinstance(values, str):
        ranges = [part.strip().upper() for part in values.split(",") if part.strip()]
    else:
        ranges = []
    allowed = {"1D", "1W", "1M", "3M", "YTD", "1Y"}
    cleaned = [r for r in ranges if r in allowed]
    if not cleaned:
        cleaned = list(fallback)
    deduped: list[str] = []
    for item in cleaned:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _ensure_brain_ranges(ranges: list[str]) -> list[str]:
    ordered: list[str] = []
    for key in BRAIN_RANGE_KEYS:
        if key not in ordered:
            ordered.append(key)
    for key in list(ranges or []):
        item = str(key).strip().upper()
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def _safe_int(value: Any, default: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), int(parsed))


def _load_queue_payload() -> list[dict[str, Any]]:
    if not PREWARM_QUEUE_PATH.exists():
        return []
    try:
        payload = json.loads(PREWARM_QUEUE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    queue = payload.get("queue", []) if isinstance(payload, dict) else []
    if not isinstance(queue, list):
        return []
    return [item for item in queue if isinstance(item, dict)]


def _merge_stats(target: dict[str, Any], stats: dict[str, Any]) -> None:
    for key in ("attempted", "live", "cache", "none", "errors"):
        target[key] = int(target.get(key, 0)) + int(stats.get(key, 0))


def _acquire_queue_lock(*, max_wait_seconds: float = 2.0, stale_seconds: float = 30.0) -> bool:
    ensure_prewarm_paths()
    deadline = time.time() + max(0.1, float(max_wait_seconds))
    lock_path = PREWARM_QUEUE_LOCK_PATH

    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - float(lock_path.stat().st_mtime)
            except OSError:
                age = 0.0
            if age > float(stale_seconds):
                try:
                    lock_path.unlink()
                except OSError:
                    pass
            time.sleep(0.02)
        except OSError:
            time.sleep(0.02)
    return False


def _release_queue_lock() -> None:
    try:
        if PREWARM_QUEUE_LOCK_PATH.exists():
            PREWARM_QUEUE_LOCK_PATH.unlink()
    except OSError:
        return


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_prewarm_paths()
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    for _ in range(5):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.05)
            continue
    # best effort fallback
    path.write_text(data, encoding="utf-8")


def should_revalidate(
    *,
    ticker: str,
    range_key: str,
    last_updated_at: datetime | None,
    now_et: datetime,
    market_bucket: Literal["market", "after_hours", "off_hours", "weekend"],
    policy_minutes: dict[str, int],
) -> bool:
    _ = str(ticker or "").strip().upper()
    _ = str(range_key or "").strip().upper()
    if last_updated_at is None:
        return True
    cadence = _safe_int(policy_minutes.get(str(market_bucket), 30), 30, minimum=1)
    elapsed_minutes = (now_et - last_updated_at).total_seconds() / 60.0
    return elapsed_minutes >= float(cadence)


def enqueue_refresh_request(
    *,
    ticker: str,
    range_keys: tuple[str, ...],
    reason: Literal["manual_refresh", "scheduled", "ticker_click", "background", "why_refresh"],
    priority: int,
) -> str:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return ""
    request = enqueue_prewarm_request(
        scope="brain",
        tickers={symbol},
        range_keys=tuple(range_keys),
        reason=str(reason),
        requested_by=f"prio_{max(0, int(priority))}",
    )
    return str(request.get("request_id", ""))


def process_refresh_batch(
    *,
    max_items: int,
    now_et: datetime,
) -> dict[str, int]:
    _ = now_et.astimezone(_ET)
    items = pop_prewarm_requests(max_items=max(1, int(max_items)))
    if not items:
        return {"processed": 0, "attempted": 0, "live": 0, "cache": 0, "none": 0, "errors": 0}
    query = MarketQueryService(cache_dir=".cache/charts", context_loader=None, cache_only=False)
    totals = {"processed": 0, "attempted": 0, "live": 0, "cache": 0, "none": 0, "errors": 0}
    for item in items:
        reason = str(item.get("reason", "manual")).strip().lower() or "manual"
        tickers = {
            str(value).strip().upper()
            for value in list(item.get("tickers", []) or [])
            if str(value).strip()
        }
        ranges = tuple(_normalize_ranges(item.get("range_keys", []), fallback=("1D", "1W")))
        if reason == "why_refresh":
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            expected_signature = str(metadata.get("why_signature", "")).strip().lower() or None
            ticker = next(iter(sorted(tickers)), "")
            stats = run_why_refresh_job(
                ticker=ticker,
                range_key=ranges[0] if ranges else "1D",
                expected_signature=expected_signature,
            )
        else:
            stats = query.revalidate_tickers(tickers=tickers, range_keys=ranges)
        totals["processed"] += 1
        for key in ("attempted", "live", "cache", "none", "errors"):
            totals[key] += int(stats.get(key, 0))
    return totals
