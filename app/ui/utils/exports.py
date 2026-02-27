from __future__ import annotations

import csv
import io
import json
from typing import Any


def to_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, indent=2, default=str).encode("utf-8")


def rows_to_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe_row = {key: _to_csv_value(row.get(key)) for key in fieldnames}
        writer.writerow(safe_row)
    return buffer.getvalue().encode("utf-8")


def _to_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)
