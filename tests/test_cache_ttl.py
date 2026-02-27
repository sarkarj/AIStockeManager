from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import parse_iso


def test_cache_set_get_and_ttl(tmp_path: Path) -> None:
    cache = DiskTTLCache(base_dir=str(tmp_path))

    key = "prices:AAPL:1h:60"
    payload = {"foo": "bar", "n": 1}
    cache.set(key=key, payload=payload, ttl_seconds=2)

    assert cache.get(key) == payload

    record_path = Path(cache.path_for_key(key))
    with record_path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    cached_at = record["cached_at"]
    future_iso = (parse_iso(cached_at) + timedelta(seconds=3)).isoformat()

    assert cache.get(key, now_iso=future_iso) is None
