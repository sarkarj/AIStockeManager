from __future__ import annotations

from copy import deepcopy

from jsonschema import validate

from app.core.llm.hub_card_generator import FORBIDDEN_TERMS, generate_hub_card
from app.core.llm.hub_card_schema import HUB_CARD_JSON_SCHEMA


class FakeClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls = 0

    def invoke_structured(self, prompt: str, json_schema: dict) -> dict:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return {"bad": "response"}


def _base_context_pack() -> dict:
    return {
        "meta": {
            "ticker": "AAPL",
            "generated_at": "2026-02-11T12:00:00-05:00",
            "interval": "1h",
            "lookback_days": 60,
            "data_quality": {
                "prices": {
                    "as_of": "2026-02-11T12:00:00-05:00",
                    "now": "2026-02-11T12:00:00-05:00",
                    "age_minutes": 0.0,
                    "stale": False,
                    "stale_minutes_threshold": 90,
                },
                "indicators": {
                    "as_of": "2026-02-11T12:00:00-05:00",
                    "now": "2026-02-11T12:00:00-05:00",
                    "age_minutes": 0.0,
                    "stale": False,
                    "stale_minutes_threshold": 90,
                },
                "overall_stale": False,
                "notes": [],
            },
        },
        "indicators": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "metrics": {
                "price_last": 320.0,
                "ema_50": 310.0,
                "sma_200": 260.0,
                "rsi_14": 58.0,
                "macd": 1.2,
                "macd_signal": 1.0,
                "stoch_k": 55.0,
                "adx_14": 24.0,
                "vroc_14": 12.0,
                "atr_pct": 2.8,
                "supertrend_dir_1D": "BULL",
                "supertrend_dir_1W": "BULL",
            },
        },
        "drl": {
            "result": {
                "action_final": "WAIT",
                "confidence_cap": 60,
                "conflicts": [],
                "decision_trace": {
                    "policy_id": "drl_v1_minimal",
                    "policy_version": "1.0.0",
                    "profile": "swing",
                    "ticker": "AAPL",
                },
            }
        },
        "news": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "items": [
                {
                    "id": "abc123def4567890",
                    "source": "stub",
                    "published_at": "2026-02-11T09:00:00-05:00",
                    "url": "https://example.com/n1",
                    "title": "AAPL update",
                    "summary": "Brief update",
                }
            ],
        },
        "macro": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "items": [
                {
                    "id": "SPY_CHANGE_1D",
                    "label": "SPY % Change (1D)",
                    "value": 0.7,
                    "source": "SPY",
                }
            ],
        },
    }


def _valid_card() -> dict:
    return {
        "meta": {
            "ticker": "AAPL",
            "generated_at": "2026-02-11T12:00:00-05:00",
            "policy_id": "drl_v1_minimal",
            "policy_version": "1.0.0",
            "profile": "swing",
            "mode": "FULL",
        },
        "summary": {
            "action_final": "WAIT",
            "confidence_cap": 60,
            "one_liner": "DRL keeps WAIT with capped confidence as technical and cross-asset signals stay mixed.",
        },
        "drivers": [
            {
                "text": "Price remains above EMA50 while RSI sits in a neutral-strength zone.",
                "citations": ["indicator:price_last", "indicator:ema_50", "indicator:rsi_14"],
            },
            {
                "text": "Recent headline flow is modest and does not override the technical posture.",
                "citations": ["news:abc123def4567890"],
            },
        ],
        "conflicts": [
            {
                "text": "Macro proxy indicates only moderate directional participation.",
                "citations": ["macro:SPY_CHANGE_1D"],
            }
        ],
        "watch": [
            {
                "text": "Watch MACD versus signal for confirmation of momentum continuation.",
                "citations": ["indicator:macd", "indicator:macd_signal"],
            },
            {
                "text": "Watch SPY one-day change for broader market pressure shifts.",
                "citations": ["macro:SPY_CHANGE_1D"],
            },
        ],
        "evidence": {
            "used_ids": [
                "indicator:price_last",
                "indicator:ema_50",
                "indicator:rsi_14",
                "indicator:macd",
                "indicator:macd_signal",
                "news:abc123def4567890",
                "macro:SPY_CHANGE_1D",
            ]
        },
    }


def _contains_forbidden(card: dict) -> bool:
    texts = [card["summary"]["one_liner"]]
    for section in ["drivers", "conflicts", "watch"]:
        texts.extend(item["text"] for item in card.get(section, []))
    text_blob = " ".join(texts).lower()
    return any(term in text_blob for term in FORBIDDEN_TERMS)


def test_generate_hub_card_accepts_valid_payload() -> None:
    context_pack = _base_context_pack()
    card = _valid_card()
    client = FakeClient([card])

    result = generate_hub_card(context_pack=context_pack, client=client, now_iso="2026-02-11T12:00:00-05:00")

    validate(instance=result, schema=HUB_CARD_JSON_SCHEMA)
    assert result["summary"]["action_final"] == "WAIT"
    assert abs(float(result["summary"]["confidence_cap"]) - 60.0) <= 0.1


def test_forbidden_terms_are_rejected_and_fallback_used() -> None:
    context_pack = _base_context_pack()
    invalid = _valid_card()
    invalid["summary"]["one_liner"] = "DRL might stay WAIT while signals stay mixed."

    client = FakeClient([invalid, invalid])
    result = generate_hub_card(context_pack=context_pack, client=client, now_iso="2026-02-11T12:00:00-05:00")

    validate(instance=result, schema=HUB_CARD_JSON_SCHEMA)
    assert result["meta"]["mode"] == "TECHNICAL_ONLY"
    assert not _contains_forbidden(result)


def test_may_forbidden_term_is_rejected_and_fallback_used() -> None:
    context_pack = _base_context_pack()
    invalid = _valid_card()
    invalid["summary"]["one_liner"] = "DRL may stay WAIT while signals stay mixed."

    client = FakeClient([invalid, invalid])
    result = generate_hub_card(context_pack=context_pack, client=client, now_iso="2026-02-11T12:00:00-05:00")

    validate(instance=result, schema=HUB_CARD_JSON_SCHEMA)
    assert result["meta"]["mode"] == "TECHNICAL_ONLY"
    assert not _contains_forbidden(result)


def test_citations_must_exist_in_evidence_catalog() -> None:
    context_pack = _base_context_pack()
    invalid = _valid_card()
    invalid["drivers"][0]["citations"] = ["indicator:not_real"]
    invalid["evidence"]["used_ids"] = ["indicator:not_real", "macro:SPY_CHANGE_1D"]

    client = FakeClient([invalid, invalid])
    result = generate_hub_card(context_pack=context_pack, client=client, now_iso="2026-02-11T12:00:00-05:00")

    validate(instance=result, schema=HUB_CARD_JSON_SCHEMA)
    assert result["summary"]["action_final"] == context_pack["drl"]["result"]["action_final"]
    assert abs(float(result["summary"]["confidence_cap"]) - float(context_pack["drl"]["result"]["confidence_cap"])) <= 0.1


def test_fallback_on_invalid_payload_is_schema_valid() -> None:
    context_pack = _base_context_pack()
    client = FakeClient([{"oops": "bad"}, {"still": "bad"}])

    result = generate_hub_card(context_pack=context_pack, client=client, now_iso="2026-02-11T12:00:00-05:00")

    validate(instance=result, schema=HUB_CARD_JSON_SCHEMA)
    assert result["summary"]["action_final"] == "WAIT"
    assert abs(float(result["summary"]["confidence_cap"]) - 60.0) <= 0.1

    result2 = deepcopy(result)
    validate(instance=result2, schema=HUB_CARD_JSON_SCHEMA)
