#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.hub_integrity.checks import verify_hub_integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline hub integrity verifier")
    parser.add_argument("--cache-dir", action="append", default=[], help="Hub cache directory to scan")
    parser.add_argument("--context-pack", action="append", default=[], help="Context pack JSON path (repeatable)")
    parser.add_argument("--metrics-json", action="append", default=[], help="Metrics JSON path (repeatable)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provided_context_paths = [Path(p) for p in args.context_pack]
    provided_metric_paths = [Path(p) for p in args.metrics_json]

    contexts, context_metrics = _discover_context_sources(provided_context_paths)
    extra_metrics = _load_metric_paths(provided_metric_paths)

    hub_entries = _discover_hub_entries(args.cache_dir, context_metrics=context_metrics)
    context_entries = _context_entries_from_contexts(contexts)

    entries = context_entries + hub_entries
    if not entries:
        print("SKIP: no hub artifacts found")
        return 0

    failed = False
    grouped: dict[str, int] = {"H1": 0, "H2": 0, "H3": 0}
    total = 0

    for entry in entries:
        total += 1
        hub = entry.get("hub")
        indicators = entry.get("indicators")
        if indicators is None and extra_metrics:
            indicators = extra_metrics
        if not isinstance(indicators, dict) or not indicators:
            print(f"[SKIP] {entry.get('id', 'unknown')} (missing indicators snapshot)")
            continue

        report = verify_hub_integrity(hub=hub, indicators=indicators)
        entry_id = entry.get("id", "unknown")
        skipped = report.get("skipped_rules", [])

        if report["ok"]:
            skip_note = f" (skipped: {', '.join(skipped)})" if skipped else ""
            print(f"[PASS] {entry_id}{skip_note}")
            continue

        failed = True
        print(f"[FAIL] {entry_id}")
        for violation in report.get("violations", []):
            rule_id = str(violation.get("rule_id", "UNKNOWN"))
            if rule_id in grouped:
                grouped[rule_id] += 1
            print(
                f"  - {rule_id}: {violation.get('expected_relation')} | "
                f"{violation.get('snippet')} | actual={violation.get('actual_values')}"
            )
        if args.fail_fast:
            break

    print("Summary:")
    print(f"  checked={total}")
    print(f"  H1_violations={grouped['H1']}")
    print(f"  H2_violations={grouped['H2']}")
    print(f"  H3_violations={grouped['H3']}")

    if failed:
        print("FAIL")
        return 1
    print("PASS")
    return 0


def _discover_context_sources(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    discovered: list[Path] = []
    discovered.extend(paths)
    discovered.extend(sorted(Path("tests/fixtures/context_packs").glob("*.json")))
    discovered.extend(sorted(Path(".cache/context_packs").glob("*.json")))
    discovered.extend(sorted(Path(".cache/replay").glob("**/*.json")))

    contexts: list[dict[str, Any]] = []
    context_metrics: dict[str, dict[str, Any]] = {}
    for path in discovered:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue

        context_pack = payload.get("context_pack") if isinstance(payload.get("context_pack"), dict) else payload
        if not isinstance(context_pack, dict):
            continue
        hub = context_pack.get("hub_card")
        indicators = context_pack.get("indicators", {}).get("metrics", {})
        if not isinstance(hub, dict) or not isinstance(indicators, dict) or not indicators:
            continue

        contexts.append(
            {
                "id": f"context:{path}",
                "hub": hub,
                "indicators": indicators,
                "path": str(path),
            }
        )
        ticker = _extract_ticker(hub=hub, context_pack=context_pack)
        if ticker:
            context_metrics[ticker] = indicators

    return contexts, context_metrics


def _context_entries_from_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return contexts


def _discover_hub_entries(cache_dirs: list[str], context_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dirs: list[Path] = []
    if cache_dirs:
        dirs.extend([Path(p) for p in cache_dirs])
    else:
        dirs.extend([Path(".cache/hub_cards"), Path(".cache/hub")])

    entries: list[dict[str, Any]] = []
    for base in dirs:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            hub = _read_json(path)
            if not isinstance(hub, dict):
                continue
            ticker = _extract_ticker(hub=hub, context_pack={})
            indicators = context_metrics.get(ticker, {}) if ticker else {}
            entries.append(
                {
                    "id": f"hub:{path}",
                    "hub": hub,
                    "indicators": indicators if indicators else None,
                    "path": str(path),
                }
            )
    return entries


def _load_metric_paths(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("metrics"), dict):
            merged.update(payload["metrics"])
        else:
            merged.update(payload)
    return merged


def _extract_ticker(hub: dict, context_pack: dict) -> str:
    ticker = str(hub.get("meta", {}).get("ticker", "")).strip().upper()
    if ticker:
        return ticker
    ticker = str(context_pack.get("meta", {}).get("ticker", "")).strip().upper()
    if ticker:
        return ticker
    return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
