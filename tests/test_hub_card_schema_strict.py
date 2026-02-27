from __future__ import annotations

import pytest
from jsonschema import ValidationError, validate

from app.core.llm.hub_card_schema import HUB_CARD_JSON_SCHEMA


def _valid_card() -> dict:
    return {
        "meta": {
            "ticker": "AAPL",
            "generated_at": "2026-02-11T12:00:00-05:00",
            "policy_id": "drl_v1_minimal",
            "policy_version": "1.0.0",
            "profile": "swing",
            "mode": "TECHNICAL_ONLY",
        },
        "summary": {
            "action_final": "WAIT",
            "confidence_cap": 60,
            "one_liner": "DRL keeps WAIT with neutral momentum and controlled volatility conditions.",
        },
        "drivers": [
            {
                "text": "Price and trend alignment are currently mixed in policy terms.",
                "citations": ["indicator:price_last"],
            },
            {
                "text": "MACD and RSI jointly indicate moderate directional conviction.",
                "citations": ["indicator:macd", "indicator:rsi_14"],
            },
        ],
        "conflicts": [],
        "watch": [
            {
                "text": "Watch RSI migration across regime thresholds.",
                "citations": ["indicator:rsi_14"],
            },
            {
                "text": "Watch MACD relative to signal for momentum persistence.",
                "citations": ["indicator:macd", "indicator:macd_signal"],
            },
        ],
        "evidence": {
            "used_ids": ["indicator:price_last", "indicator:macd", "indicator:rsi_14", "indicator:macd_signal"]
        },
    }


def test_schema_rejects_additional_properties() -> None:
    card = _valid_card()
    card["meta"]["unexpected"] = "x"

    with pytest.raises(ValidationError):
        validate(instance=card, schema=HUB_CARD_JSON_SCHEMA)


def test_schema_rejects_missing_required_fields() -> None:
    card = _valid_card()
    del card["summary"]["one_liner"]

    with pytest.raises(ValidationError):
        validate(instance=card, schema=HUB_CARD_JSON_SCHEMA)
