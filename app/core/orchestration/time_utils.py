from __future__ import annotations

from datetime import datetime, timezone


def parse_iso(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def minutes_between(a_iso: str, b_iso: str) -> float:
    a_dt = parse_iso(a_iso)
    b_dt = parse_iso(b_iso)
    return (b_dt - a_dt).total_seconds() / 60.0


def is_stale(as_of_iso: str, now_iso: str, stale_minutes: int) -> bool:
    return minutes_between(as_of_iso, now_iso) > float(stale_minutes)
