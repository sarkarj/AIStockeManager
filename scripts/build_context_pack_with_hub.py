from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import now_iso


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic context packs with hub cards")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-days", type=int, default=60)
    args = parser.parse_args()

    cache = DiskTTLCache(base_dir=".cache")
    provider = SampleMarketDataProvider()
    generated_at = now_iso()

    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"
    out_dir = Path(".cache") / "context_packs"
    out_dir.mkdir(parents=True, exist_ok=True)

    bedrock_config = {
        "region": os.getenv("AWS_REGION", ""),
        "model_id": os.getenv("BEDROCK_MODEL_ID", ""),
    }

    for ticker in args.tickers:
        pack = build_context_pack(
            ticker=ticker,
            now_iso=generated_at,
            provider=provider,
            cache=cache,
            policy_path=str(policy_path),
            lookback_days=args.lookback_days,
            interval=args.interval,
            generate_hub_card=True,
            bedrock_config=bedrock_config,
        )

        file_ts = _safe_filename_ts(pack["meta"]["generated_at"])
        out_path = out_dir / f"{ticker}-{file_ts}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pack, f, ensure_ascii=True, indent=2, sort_keys=True)

        drl_result = pack["drl"]["result"]
        hub_mode = pack.get("hub_card", {}).get("meta", {}).get("mode", "DEGRADED")
        print(f"{ticker} | action={drl_result['action_final']} | conf={drl_result['confidence_cap']} | hub_mode={hub_mode}")


def _safe_filename_ts(ts: str) -> str:
    return ts.replace(":", "-")


if __name__ == "__main__":
    main()
