#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.context_pack.hub_generator import HUB_FORBIDDEN_TERMS
from app.core.hub_integrity.checks import verify_hub_integrity

HEDGE_WORDS = {"maybe", "might", "could", "possibly", "likely"}


def main() -> int:
    entries = _discover_entries()
    if not entries:
        print("SKIP: no hub artifacts found")
        return 0

    failed = False
    for entry in entries:
        hub = entry["hub"]
        indicators = entry.get("indicators")
        reasons: list[str] = []

        base_report = verify_hub_integrity(hub=hub, indicators=indicators if isinstance(indicators, dict) else {})
        if not base_report["ok"]:
            for violation in base_report["violations"]:
                rid = str(violation.get("rule_id"))
                if rid == "H1":
                    reasons.append("HUB_FORBIDDEN_TERM")
                elif rid == "H2":
                    reasons.append("HUB_DUP_CITATIONS")
                elif rid == "H3":
                    reasons.append("HUB_CONTRADICTION")

        if _has_hedge_words(hub):
            reasons.append("HUB_HEDGE_WORD")
        if _has_keyword_contradiction(hub):
            reasons.append("HUB_CONTRADICTION")

        reason_set = sorted(set(reasons))
        if reason_set:
            failed = True
            print(f"[FAIL] {entry['id']} :: {', '.join(reason_set)}")
        else:
            print(f"[PASS] {entry['id']}")

    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


def _discover_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    paths = list(Path(".cache/context_packs").glob("*.json"))
    paths += list(Path(".cache/replay").glob("**/*.json"))
    paths += list(Path("tests/fixtures/context_packs").glob("*.json"))
    for path in sorted(paths):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        context_pack = payload.get("context_pack") if isinstance(payload.get("context_pack"), dict) else payload
        if not isinstance(context_pack, dict):
            continue
        hub = context_pack.get("hub_card")
        if not isinstance(hub, dict):
            continue
        indicators = context_pack.get("indicators", {}).get("metrics", {})
        entries.append(
            {
                "id": f"context:{path}",
                "hub": hub,
                "indicators": indicators if isinstance(indicators, dict) else {},
            }
        )

    for path in sorted(Path(".cache/hub_cards").glob("*.json")):
        hub = _read_json(path)
        if not isinstance(hub, dict):
            continue
        entries.append({"id": f"hub:{path}", "hub": hub, "indicators": {}})
    return entries


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_text(node: Any) -> str:
    chunks: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            chunks.append(_extract_text(value))
    elif isinstance(node, list):
        for item in node:
            chunks.append(_extract_text(item))
    elif isinstance(node, str):
        chunks.append(node)
    return " ".join(chunk for chunk in chunks if chunk)


def _has_hedge_words(hub: dict[str, Any]) -> bool:
    text = _extract_text(hub).lower()
    banned = set(word.lower() for word in HUB_FORBIDDEN_TERMS) | HEDGE_WORDS
    for token in banned:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


def _has_keyword_contradiction(hub: dict[str, Any]) -> bool:
    text = _extract_text(hub).lower()
    bullish = any(token in text for token in ["bullish", "buy"])
    bearish = any(token in text for token in ["bearish", "sell"])
    return bullish and bearish


if __name__ == "__main__":
    raise SystemExit(main())
