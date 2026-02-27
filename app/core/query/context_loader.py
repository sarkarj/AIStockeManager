from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import now_iso

_POLICY_PATH = Path(__file__).resolve().parents[1] / "drl" / "policies" / "drl_policy.yaml"


def load_context_pack_for_query(
    *,
    ticker: str,
    generate_hub_card: bool = False,
    interval: str = "1h",
    lookback_days: int = 60,
    hub_request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    provider = SampleMarketDataProvider()
    cache = DiskTTLCache(base_dir=".cache")

    bedrock_region = os.getenv("AWS_REGION", "") or os.getenv("AWS_DEFAULT_REGION", "")
    bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "")
    bedrock_claude_model_id = os.getenv("BEDROCK_LLM_ID_CLAUDE", "")
    bedrock_openai_model_id = os.getenv("BEDROCK_LLM_ID_OPENAI", "")
    bedrock_config = None
    if bedrock_region.strip() and (
        bedrock_model_id.strip() or bedrock_claude_model_id.strip() or bedrock_openai_model_id.strip()
    ):
        bedrock_config = {
            "region": bedrock_region.strip(),
            "model_id": bedrock_model_id.strip(),
            "claude_model_id": bedrock_claude_model_id.strip(),
            "openai_model_id": bedrock_openai_model_id.strip(),
        }

    return build_context_pack(
        ticker=str(ticker).strip().upper(),
        now_iso=now_iso(),
        provider=provider,
        cache=cache,
        policy_path=str(_POLICY_PATH),
        lookback_days=int(lookback_days),
        interval=str(interval),
        generate_hub_card=bool(generate_hub_card),
        bedrock_config=bedrock_config,
        hub_request_timeout_seconds=hub_request_timeout_seconds,
    )
