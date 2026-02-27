from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.context_pack.hub_generator import generate_hub_for_context_pack
from app.core.drl.drl_engine import evaluate_drl
from app.core.indicators.compute_indicators import compute_required_metrics_from_prices
from app.core.orchestration.freshness import freshness_status

if TYPE_CHECKING:
    from app.core.marketdata.provider import MarketDataProvider
    from app.core.orchestration.cache import DiskTTLCache


def build_context_pack(
    ticker: str,
    now_iso: str,
    provider: "MarketDataProvider",
    cache: "DiskTTLCache",
    policy_path: str,
    lookback_days: int = 60,
    interval: str = "1h",
    ttl_seconds_prices: int = 3600,
    generate_hub_card: bool = False,
    bedrock_config: dict | None = None,
    hub_request_timeout_seconds: float | None = None,
) -> dict:
    cache_key = f"prices:{ticker}:{interval}:{lookback_days}"
    prices_payload = cache.get(cache_key, now_iso=now_iso)

    if prices_payload is None:
        prices_payload = provider.get_ohlcv(ticker=ticker, interval=interval, lookback_days=lookback_days)
        cache.set(cache_key, prices_payload, ttl_seconds=ttl_seconds_prices)

    prices = {
        "as_of": str(prices_payload["as_of"]),
        "bars": [_normalize_bar(bar) for bar in prices_payload.get("bars", [])],
    }

    metrics = compute_required_metrics_from_prices(prices)

    # TEMPORARY Phase 2 proxy until real Supertrend is added.
    metrics["supertrend_dir_1D"] = "BULL" if metrics["price_last"] > metrics["ema_50"] else "BEAR"
    # TEMPORARY Phase 2 proxy until real Supertrend is added.
    metrics["supertrend_dir_1W"] = "BULL" if metrics["price_last"] > metrics["sma_200"] else "BEAR"

    indicators = {
        "as_of": prices["as_of"],
        "metrics": metrics,
    }

    drl_inputs = {
        "ticker": ticker,
        "as_of": indicators["as_of"],
        **metrics,
    }
    drl_result = evaluate_drl(policy_path=policy_path, inputs=drl_inputs, now_iso=now_iso)

    prices_freshness = freshness_status(prices["as_of"], now_iso, stale_minutes=90)
    indicators_freshness = freshness_status(indicators["as_of"], now_iso, stale_minutes=90)

    notes: list[str] = []
    overall_stale = False
    if prices_freshness["stale"]:
        overall_stale = True
        notes.append("Prices are stale (>90 minutes).")
    if indicators_freshness["stale"]:
        overall_stale = True
        notes.append("Indicators are stale (>90 minutes).")

    bars = prices.get("bars", [])
    if len(bars) < 10:
        notes.append("INSUFFICIENT_BARS")
    if _has_suspect_series(bars):
        notes.append("SUSPECT_SERIES")

    context_pack = {
        "meta": {
            "ticker": ticker,
            "generated_at": now_iso,
            "interval": interval,
            "lookback_days": lookback_days,
            "data_quality": {
                "prices": prices_freshness,
                "indicators": indicators_freshness,
                "overall_stale": overall_stale,
                "notes": notes,
            },
        },
        "prices": prices,
        "indicators": indicators,
        "drl": {
            "result": drl_result,
            "decision_trace": drl_result["decision_trace"],
        },
    }

    if generate_hub_card:
        hub_result = generate_hub_for_context_pack(
            context_pack=context_pack,
            now_iso=now_iso,
            bedrock_config=bedrock_config,
            request_timeout_seconds=hub_request_timeout_seconds,
        )
        if isinstance(hub_result.hub_card, dict):
            context_pack["hub_card"] = hub_result.hub_card

        context_pack["meta"]["hub"] = {
            "status": hub_result.status,
            "mode": hub_result.mode,
            "reason": hub_result.reason,
            "cache_path": hub_result.cache_path,
            "from_cache": bool(hub_result.from_cache),
            "hub_valid": bool(hub_result.hub_valid),
            "llm_usage": hub_result.llm_usage if isinstance(hub_result.llm_usage, dict) else {},
        }
        if hub_result.reason:
            notes.append(f"HUB: {hub_result.reason}")
    else:
        context_pack["meta"]["hub"] = {
            "status": "missing",
            "mode": "DEGRADED",
            "reason": "Hub generation disabled",
            "cache_path": None,
            "from_cache": False,
            "hub_valid": False,
        }
    return context_pack


def _normalize_bar(bar: dict) -> dict:
    return {
        "ts": str(bar["ts"]),
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": None if bar.get("volume") is None else float(bar["volume"]),
    }


def _has_suspect_series(bars: list[dict]) -> bool:
    if len(bars) < 10:
        return False
    closes: list[float] = []
    for bar in bars:
        try:
            closes.append(float(bar.get("close", 0.0)))
        except (TypeError, ValueError):
            continue
    if len(closes) < 10:
        return False
    repeated = 0
    for idx in range(1, len(closes)):
        if closes[idx] == closes[idx - 1]:
            repeated += 1
    return (repeated / max(1, len(closes) - 1)) >= 0.8
