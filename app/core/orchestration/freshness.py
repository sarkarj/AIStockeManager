from __future__ import annotations

from app.core.orchestration import time_utils


def freshness_status(as_of_iso: str, now_iso: str, stale_minutes: int = 90) -> dict:
    age_minutes = time_utils.minutes_between(as_of_iso, now_iso)
    stale = time_utils.is_stale(as_of_iso, now_iso, stale_minutes)
    return {
        "as_of": as_of_iso,
        "now": now_iso,
        "age_minutes": age_minutes,
        "stale": stale,
        "stale_minutes_threshold": stale_minutes,
    }
