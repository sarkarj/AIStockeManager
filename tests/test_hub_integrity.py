from __future__ import annotations

from app.core.hub_integrity.checks import verify_hub_integrity


def _hub_base(text: str = "Deterministic summary text.") -> dict:
    return {
        "meta": {"ticker": "AAPL"},
        "summary": {"one_liner": text},
        "drivers": [
            {"text": "Price above EMA50 and MACD bullish.", "citations": ["indicator:price_last", "indicator:ema_50"]},
            {"text": "RSI is neutral and ADX is moderate.", "citations": ["indicator:rsi_14", "indicator:adx_14"]},
        ],
        "conflicts": [],
        "watch": [
            {"text": "Watch MACD signal relationship.", "citations": ["indicator:macd", "indicator:macd_signal"]},
            {"text": "Watch ATR expansion.", "citations": ["indicator:atr_pct"]},
        ],
        "evidence": {
            "used_ids": [
                "indicator:price_last",
                "indicator:ema_50",
                "indicator:rsi_14",
                "indicator:adx_14",
                "indicator:macd",
                "indicator:macd_signal",
                "indicator:atr_pct",
            ]
        },
    }


def _indicators() -> dict:
    return {
        "price_last": 120.0,
        "ema_50": 110.0,
        "sma_200": 100.0,
        "rsi_14": 55.0,
        "macd": 1.2,
        "macd_signal": 0.8,
        "adx_14": 24.0,
    }


def test_h1_hedge_word_fails() -> None:
    hub = _hub_base(text="This could break out soon.")
    report = verify_hub_integrity(hub=hub, indicators=_indicators())
    assert report["ok"] is False
    assert any(v["rule_id"] == "H1" for v in report["violations"])


def test_h2_duplicate_and_concatenated_citations_fail() -> None:
    hub = _hub_base()
    hub["drivers"][0]["citations"] = ["Refs: Citations: indicator:adx_14, indicator:adx_14"]
    hub["watch"][0]["citations"] = ["indicator:adx_14indicator:rsi_14"]

    report = verify_hub_integrity(hub=hub, indicators=_indicators())
    assert report["ok"] is False
    assert any(v["rule_id"] == "H2" for v in report["violations"])
    assert any("Refs: Citations:" in str(v.get("snippet", "")) for v in report["violations"])


def test_h3_numeric_contradiction_fails() -> None:
    hub = _hub_base()
    hub["drivers"][0]["text"] = "The setup confirms price above EMA50."

    broken_indicators = _indicators()
    broken_indicators["price_last"] = 100.0
    broken_indicators["ema_50"] = 110.0

    report = verify_hub_integrity(hub=hub, indicators=broken_indicators)
    assert report["ok"] is False
    assert any(v["rule_id"] == "H3" for v in report["violations"])
