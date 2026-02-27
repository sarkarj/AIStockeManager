from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.orchestration import time_utils


class DiskTTLCache:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return str(self.base_dir / f"{digest}.json")

    def get(self, key: str, now_iso: str | None = None) -> dict[str, Any] | None:
        path = Path(self.path_for_key(key))
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        cached_at = record.get("cached_at")
        ttl_seconds = record.get("ttl_seconds")
        payload = record.get("payload")

        if not isinstance(cached_at, str) or not isinstance(ttl_seconds, int) or not isinstance(payload, dict):
            return None

        effective_now_iso = now_iso or time_utils.now_iso()
        age_seconds = (time_utils.parse_iso(effective_now_iso) - time_utils.parse_iso(cached_at)).total_seconds()
        if age_seconds > ttl_seconds:
            return None

        return payload

    def set(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        path = Path(self.path_for_key(key))
        record = {
            "cached_at": time_utils.now_iso(),
            "ttl_seconds": int(ttl_seconds),
            "payload": payload,
        }
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=True, sort_keys=True)
        except OSError:
            # Cache write failures should not break runtime flows.
            return
