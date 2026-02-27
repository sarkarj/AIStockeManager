from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from app.core.context_pack import hub_generator
from app.ui.components import brain
from app.ui.viewmodels.brain_vm import build_brain_view_model


class FakeBedrockClient:
    def __init__(
        self,
        region: str,
        model_id: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
        request_timeout_seconds: float | None = None,
    ):
        self.region = region
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds

    def invoke_structured(self, prompt: str, json_schema: dict) -> dict:
        return _valid_card(model_id=self.model_id)

    def get_last_usage(self) -> dict:
        return {
            "model_id": self.model_id,
            "transport": "fake",
            "latency_ms": 1.0,
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }


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
        "prices": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "bars": [
                {
                    "ts": "2026-02-11T11:00:00-05:00",
                    "open": 320.0,
                    "high": 322.0,
                    "low": 319.0,
                    "close": 321.0,
                    "volume": 1000.0,
                }
            ],
        },
        "indicators": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "metrics": {
                "price_last": 321.0,
                "ema_50": 315.0,
                "sma_200": 280.0,
                "rsi_14": 57.0,
                "macd": 1.2,
                "macd_signal": 1.0,
                "stoch_k": 55.0,
                "adx_14": 24.0,
                "vroc_14": 12.0,
                "atr_pct": 2.2,
                "supertrend_dir_1D": "BULL",
                "supertrend_dir_1W": "BULL",
            },
        },
        "drl": {
            "result": {
                "action_final": "WAIT",
                "confidence_cap": 60,
                "gates_triggered": [],
                "conflicts": [],
                "decision_trace": {
                    "policy_id": "drl_v1_minimal",
                    "policy_version": "1.0.0",
                    "profile": "swing",
                    "ticker": "AAPL",
                },
            }
        },
    }


def _valid_card(model_id: str) -> dict:
    model_tag = "Claude" if "claude" in model_id.lower() else "GPT"
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
            "one_liner": f"{model_tag} aligns with DRL WAIT while technical evidence remains mixed but stable.",
        },
        "drivers": [
            {
                "text": f"{model_tag} notes price above EMA50 with neutral RSI support.",
                "citations": ["indicator:price_last", "indicator:ema_50", "indicator:rsi_14"],
            },
            {
                "text": f"{model_tag} confirms MACD remains constructive relative to signal.",
                "citations": ["indicator:macd", "indicator:macd_signal"],
            },
        ],
        "conflicts": [
            {
                "text": "No major external conflict is present in the current technical-only context.",
                "citations": ["indicator:adx_14"],
            }
        ],
        "watch": [
            {
                "text": "Watch RSI drift for a move into stronger momentum territory.",
                "citations": ["indicator:rsi_14"],
            },
            {
                "text": "Watch ATR percentage for volatility expansion risk.",
                "citations": ["indicator:atr_pct"],
            },
        ],
        "evidence": {
            "used_ids": [
                "indicator:price_last",
                "indicator:ema_50",
                "indicator:rsi_14",
                "indicator:macd",
                "indicator:macd_signal",
                "indicator:adx_14",
                "indicator:atr_pct",
            ]
        },
    }


def test_hub_present_when_llm_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hub_generator, "BedrockLLMClient", FakeBedrockClient)
    monkeypatch.setattr(hub_generator, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(
        hub_generator,
        "_hub_cache_path",
        lambda ticker, as_of_iso: str(tmp_path / f"{ticker}-{as_of_iso}.json"),
    )

    pack = _base_context_pack()
    result = hub_generator.generate_hub_for_context_pack(
        context_pack=pack,
        now_iso="2026-02-11T12:00:00-05:00",
        bedrock_config={
            "region": "us-east-1",
            "claude_model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "openai_model_id": "openai.gpt-oss-120b-1:0",
        },
    )

    assert result.status == "present"
    assert isinstance(result.hub_card, dict)
    assert "Claude and GPT show" in result.hub_card["summary"]["one_liner"]
    assert isinstance(result.llm_usage, dict)
    assert int(result.llm_usage.get("total_tokens", 0)) > 0

    view_pack = deepcopy(pack)
    view_pack["hub_card"] = result.hub_card
    view_pack["meta"]["hub"] = {"status": result.status, "mode": result.mode}
    vm = build_brain_view_model(view_pack)

    assert vm["one_liner"] == result.hub_card["summary"]["one_liner"]
    assert vm["badges"]["grounded"] is True


def test_hub_missing_when_env_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(hub_generator, "has_aws_credentials", lambda: False)

    pack = _base_context_pack()
    result = hub_generator.generate_hub_for_context_pack(
        context_pack=pack,
        now_iso="2026-02-11T12:00:00-05:00",
        bedrock_config={"region": "", "model_id": ""},
    )

    assert result.status == "missing"
    assert result.hub_card is None

    vm = build_brain_view_model(pack)
    assert vm["badges"]["grounded"] is False

    monkeypatch.setattr(brain, "hub_is_llm_configured", lambda: False)
    fallback = brain.build_why_fallback(context_pack=pack, vm=vm)
    assert "LLM not configured" in fallback["reason"]


def test_hub_postprocess_removes_newlines_and_duplicate_drivers(monkeypatch, tmp_path: Path) -> None:
    raw_card = _valid_card(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    raw_card["drivers"] = [
        {
            "text": "Trend line one\\nwith break",
            "citations": ["indicator:price_last", "indicator:ema_50"],
        },
        {
            "text": "Trend line one\\nwith break",
            "citations": ["indicator:price_last", "indicator:ema_50"],
        },
    ]

    monkeypatch.setattr(hub_generator, "BedrockLLMClient", FakeBedrockClient)
    monkeypatch.setattr(hub_generator, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(
        hub_generator,
        "generate_hub_card",
        lambda context_pack, client, now_iso: raw_card,
    )
    monkeypatch.setattr(
        hub_generator,
        "_hub_cache_path",
        lambda ticker, as_of_iso: str(tmp_path / f"{ticker}-{as_of_iso}.json"),
    )

    pack = _base_context_pack()
    result = hub_generator.generate_hub_for_context_pack(
        context_pack=pack,
        now_iso="2026-02-11T12:00:00-05:00",
        bedrock_config={"region": "us-east-1", "model_id": "anthropic.claude-3-haiku-20240307-v1:0"},
    )

    assert result.status == "present"
    hub = result.hub_card or {}
    texts = [item.get("text", "") for item in hub.get("drivers", [])]
    assert len(texts) == len(set(texts))
    assert all("\\n" not in str(text) for text in texts)


def test_content_reference_formatter_is_readable() -> None:
    rendered = brain._render_content_references(
        ["indicator:adx_14", "indicator:rsi_14", "indicator:adx_14", "macro:SPY_CHANGE_1D"]
    )
    assert "adx_14, rsi_14, macro:SPY_CHANGE_1D" in rendered
    assert "adx_14adx_14" not in rendered
