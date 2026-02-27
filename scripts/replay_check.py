from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.orchestration.time_utils import now_iso
from app.core.replay.artifact_store import list_artifacts, load_artifact
from app.core.replay.replay_engine import replay_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-check saved artifacts against current DRL")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--latest", dest="latest", action="store_true")
    parser.add_argument("--all", dest="latest", action="store_false")
    parser.set_defaults(latest=True)
    args = parser.parse_args()

    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    artifact_paths: list[str] = []
    if args.path:
        artifact_paths = [args.path]
    else:
        if not args.ticker:
            print("ERROR: provide --path or --ticker")
            raise SystemExit(2)

        paths = list_artifacts(args.ticker)
        if not paths:
            print(f"No artifacts found for ticker: {args.ticker}")
            raise SystemExit(1)

        artifact_paths = [paths[0]] if args.latest else paths

    any_mismatch = False
    for artifact_path in artifact_paths:
        artifact = load_artifact(artifact_path)
        result = replay_artifact(artifact=artifact, policy_path=str(policy_path), now_iso=now_iso())

        if result.get("ok"):
            print(f"OK | {artifact_path}")
        else:
            any_mismatch = True
            print(f"MISMATCH | {artifact_path}")
            print(f"expected={result.get('expected')}")
            print(f"actual={result.get('actual')}")
            print(f"diff={result.get('diff')}")
            if result.get("policy_mismatch"):
                print(
                    "policy_mismatch="
                    f"artifact:{result.get('artifact_policy_hash')} current:{result.get('current_policy_hash')}"
                )

    if any_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
