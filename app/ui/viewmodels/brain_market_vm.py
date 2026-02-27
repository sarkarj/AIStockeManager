from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable
from typing import Any

from app.core.marketdata.chart_fetcher import Bar, ChartFetcher, ChartSeries
from app.core.marketdata.price_sanity import reconcile_price_last
from app.core.orchestration.time_utils import now_iso, parse_iso
from app.ui.components.trust_badges import compute_brain_trust_state


@dataclass(frozen=True)
class QuoteStats:
    open: float | None
    volume: int | None
    day_low: float | None
    day_high: float | None
    year_low: float | None
    year_high: float | None
    source_range_day: str
    source_range_year: str
    notes: list[str]


@dataclass(frozen=True)
class Gauge:
    label: str
    value: str
    score: float
    tone: str  # good | neutral | bad


@dataclass(frozen=True)
class BrainMarketVM:
    ticker: str
    price_display: float | None
    as_of_price: datetime | None
    chart_series: ChartSeries
    quote: QuoteStats
    gauges: list[Gauge]
    badge_state: dict[str, bool]
    why_block: dict[str, Any]
    chart_source_range: str
    gauge_basis_note: str


def build_brain_market_vm(
    context_pack: dict,
    selected_range_key: str,
    advanced_source_range: str | None = None,
    chart_fetcher: ChartFetcher | None = None,
    series_resolver: Callable[[str], ChartSeries] | None = None,
    badge_state: dict[str, bool] | None = None,
    why_block: dict[str, Any] | None = None,
) -> BrainMarketVM:
    ticker = str(context_pack.get("meta", {}).get("ticker", "")).strip().upper()
    fetcher = chart_fetcher or ChartFetcher(cache_dir=".cache/charts")

    source_range = _source_range_for_key(selected_range_key, advanced_source_range=advanced_source_range)
    selected_series = _safe_fetch_series(
        fetcher=fetcher,
        ticker=ticker,
        range_key=source_range,
        series_resolver=series_resolver,
    )
    day_series = _safe_fetch_series(
        fetcher=fetcher,
        ticker=ticker,
        range_key="1D",
        series_resolver=series_resolver,
    )
    year_series = _safe_fetch_series(
        fetcher=fetcher,
        ticker=ticker,
        range_key="1Y",
        series_resolver=series_resolver,
    )

    quote, primary_close = _build_quote_stats(day_series=day_series, year_series=year_series)
    fallback_close = _last_close(selected_series.bars) if source_range != "1D" else None
    indicator_price_last = _to_float_or_none(context_pack.get("indicators", {}).get("metrics", {}).get("price_last"))
    sanity = reconcile_price_last(
        ticker=ticker,
        indicator_price_last=indicator_price_last,
        primary_series_close=primary_close,
        fallback_series_close=fallback_close,
    )
    price_display = sanity.display_price

    metrics = context_pack.get("indicators", {}).get("metrics", {})
    gauges = _build_gauges(metrics if isinstance(metrics, dict) else {})
    notes = list(quote.notes)
    for flag in sorted(sanity.quality_flags):
        if flag not in notes:
            notes.append(flag)
    quote = QuoteStats(
        open=quote.open,
        volume=quote.volume,
        day_low=quote.day_low,
        day_high=quote.day_high,
        year_low=quote.year_low,
        year_high=quote.year_high,
        source_range_day=quote.source_range_day,
        source_range_year=quote.source_range_year,
        notes=notes,
    )

    computed_badges = badge_state or compute_brain_trust_state(
        context_pack=context_pack,
        chart_series={
            "source": selected_series.source,
            "flags": sorted(selected_series.quality_flags),
            "diagnostics": {"error": selected_series.error},
            "point_count": len(selected_series.bars),
        },
        market_data_provider_up=len(selected_series.bars) > 0,
    )
    computed_why = why_block or _build_default_why_block(context_pack)

    return BrainMarketVM(
        ticker=ticker,
        price_display=price_display,
        as_of_price=day_series.as_of if primary_close is not None else None,
        chart_series=selected_series,
        quote=quote,
        gauges=gauges,
        badge_state=computed_badges,
        why_block=computed_why,
        chart_source_range=source_range,
        gauge_basis_note="Gauges based on 1D indicators",
    )


def _source_range_for_key(range_key: str, advanced_source_range: str | None = None) -> str:
    key = str(range_key).upper()
    if key == "ADVANCED":
        advanced = str(advanced_source_range or "3M").upper()
        return advanced if advanced in {"1D", "1W", "1M", "3M", "YTD", "1Y"} else "3M"
    allowed = {"1D", "1W", "1M", "3M", "YTD", "1Y"}
    return key if key in allowed else "3M"


def _safe_fetch_series(
    fetcher: ChartFetcher,
    ticker: str,
    range_key: str,
    series_resolver: Callable[[str], ChartSeries] | None = None,
) -> ChartSeries:
    try:
        if callable(series_resolver):
            resolved = series_resolver(str(range_key).upper())
            if isinstance(resolved, ChartSeries):
                return resolved
        return fetcher.fetch_chart_series(ticker=ticker, range_key=range_key)
    except Exception as exc:
        timestamp = parse_iso(now_iso())
        return ChartSeries(
            bars=[],
            as_of=timestamp,
            source="none",
            error=f"fetch_failed:{str(exc)}"[:160],
            quality_flags={"MISSING"},
            cache_path="",
            cache_age_minutes=None,
            cache_hit=False,
            stale_cache=False,
            attempts=0,
        )


def _build_quote_stats(day_series: ChartSeries, year_series: ChartSeries) -> tuple[QuoteStats, float | None]:
    notes: list[str] = []
    day_bars = day_series.bars
    year_bars = year_series.bars

    if not day_bars:
        notes.append("DAY_RANGE_MISSING")
    if not year_bars:
        notes.append("YEAR_RANGE_MISSING")

    open_px: float | None = None
    day_low: float | None = None
    day_high: float | None = None
    day_volume: int | None = None
    price_close: float | None = None
    if day_bars:
        sorted_day = sorted(day_bars, key=lambda b: b.ts)
        open_px = float(sorted_day[0].open)
        day_low = min(float(bar.low) for bar in sorted_day)
        day_high = max(float(bar.high) for bar in sorted_day)
        volumes = [bar.volume for bar in sorted_day if bar.volume is not None]
        day_volume = int(sum(float(v) for v in volumes)) if volumes else None
        price_close = float(sorted_day[-1].close)

    year_low: float | None = None
    year_high: float | None = None
    if year_bars:
        sorted_year = sorted(year_bars, key=lambda b: b.ts)
        year_low = min(float(bar.low) for bar in sorted_year)
        year_high = max(float(bar.high) for bar in sorted_year)

    return (
        QuoteStats(
            open=open_px,
            volume=day_volume,
            day_low=day_low,
            day_high=day_high,
            year_low=year_low,
            year_high=year_high,
            source_range_day="1D",
            source_range_year="1Y",
            notes=notes,
        ),
        price_close,
    )


def _build_gauges(metrics: dict[str, Any]) -> list[Gauge]:
    return [
        _build_trend_gauge(metrics),
        _build_momentum_gauge(metrics),
        _build_risk_gauge(metrics),
        _build_strength_gauge(metrics),
    ]


def _build_trend_gauge(metrics: dict[str, Any]) -> Gauge:
    price_last = _to_float_or_none(metrics.get("price_last"))
    ema_50 = _to_float_or_none(metrics.get("ema_50"))
    sma_200 = _to_float_or_none(metrics.get("sma_200"))
    st_1d = str(metrics.get("supertrend_dir_1D", "")).upper()
    if price_last is None or ema_50 is None or sma_200 is None:
        return Gauge(label="TREND", value="—", score=50.0, tone="neutral")

    favorable = (price_last >= ema_50 and ema_50 >= sma_200) or st_1d == "BULL"
    unfavorable = (price_last < ema_50 and ema_50 < sma_200) or st_1d == "BEAR"
    if favorable and not unfavorable:
        return Gauge(label="TREND", value="BULLISH", score=82.0, tone="good")
    if unfavorable and not favorable:
        return Gauge(label="TREND", value="BEARISH", score=18.0, tone="bad")
    return Gauge(label="TREND", value="NEUTRAL", score=50.0, tone="neutral")


def _build_momentum_gauge(metrics: dict[str, Any]) -> Gauge:
    macd = _to_float_or_none(metrics.get("macd"))
    macd_signal = _to_float_or_none(metrics.get("macd_signal"))
    rsi_14 = _to_float_or_none(metrics.get("rsi_14"))
    if macd is None or macd_signal is None or rsi_14 is None:
        return Gauge(label="MOMENTUM", value="—", score=50.0, tone="neutral")

    favorable = (rsi_14 >= 55.0) or (macd > macd_signal and macd > 0.0)
    unfavorable = (rsi_14 <= 45.0) or (macd < macd_signal and macd < 0.0)
    if favorable and not unfavorable:
        return Gauge(label="MOMENTUM", value="BULLISH", score=80.0, tone="good")
    if unfavorable and not favorable:
        return Gauge(label="MOMENTUM", value="BEARISH", score=20.0, tone="bad")
    return Gauge(label="MOMENTUM", value="NEUTRAL", score=50.0, tone="neutral")


def _build_risk_gauge(metrics: dict[str, Any]) -> Gauge:
    atr_pct = _to_float_or_none(metrics.get("atr_pct"))
    adx_14 = _to_float_or_none(metrics.get("adx_14"))
    rsi_14 = _to_float_or_none(metrics.get("rsi_14"))
    if atr_pct is None:
        return Gauge(label="RISK", value="—", score=50.0, tone="neutral")

    if atr_pct >= 1.5 or ((adx_14 is not None and adx_14 >= 35.0) and (rsi_14 is not None and rsi_14 <= 30.0)):
        return Gauge(label="RISK", value="HIGH", score=90.0, tone="bad")
    if atr_pct <= 0.8:
        return Gauge(label="RISK", value="LOW", score=20.0, tone="good")
    return Gauge(label="RISK", value="MED", score=55.0, tone="neutral")


def _build_strength_gauge(metrics: dict[str, Any]) -> Gauge:
    adx_14 = _to_float_or_none(metrics.get("adx_14"))
    vroc_14 = _to_float_or_none(metrics.get("vroc_14"))
    if adx_14 is None or vroc_14 is None:
        return Gauge(label="STRENGTH", value="—", score=50.0, tone="neutral")

    favorable = adx_14 >= 25.0 and vroc_14 > 0.0
    unfavorable = adx_14 < 15.0 or vroc_14 < 0.0
    if favorable and not unfavorable:
        return Gauge(label="STRENGTH", value="STRONG", score=82.0, tone="good")
    if unfavorable and not favorable:
        return Gauge(label="STRENGTH", value="WEAK", score=22.0, tone="bad")
    return Gauge(label="STRENGTH", value="OK", score=52.0, tone="neutral")


def _build_default_why_block(context_pack: dict) -> dict[str, Any]:
    hub_card = context_pack.get("hub_card")
    if isinstance(hub_card, dict):
        summary = hub_card.get("summary", {}) if isinstance(hub_card.get("summary"), dict) else {}
        one_liner = str(summary.get("one_liner", "")).strip()
        if one_liner:
            return {
                "mode": "HUB",
                "one_liner": one_liner,
                "drivers": _as_dict_list(hub_card.get("drivers", [])),
                "conflicts": _as_dict_list(hub_card.get("conflicts", [])),
                "watch": _as_dict_list(hub_card.get("watch", [])),
            }

    drl_result = context_pack.get("drl", {}).get("result", {}) if isinstance(context_pack.get("drl"), dict) else {}
    trace = drl_result.get("decision_trace", {}) if isinstance(drl_result.get("decision_trace"), dict) else {}
    return {
        "mode": "FALLBACK",
        "title": "The Why (Deterministic fallback)",
        "reason": _fallback_reason(context_pack),
        "action_final": str(drl_result.get("action_final", "WAIT")),
        "confidence_cap": _to_float_or_none(drl_result.get("confidence_cap")) or 0.0,
        "drivers": _trace_driver_lines(trace=trace, drl_result=drl_result),
        "gates": [str(x) for x in drl_result.get("gates_triggered", [])],
        "conflicts": [str(x) for x in drl_result.get("conflicts", [])],
    }


def _fallback_reason(context_pack: dict) -> str:
    hub_reason = str(context_pack.get("meta", {}).get("hub", {}).get("reason", "")).strip()
    if hub_reason:
        return hub_reason
    return "Context pack missing hub artifact"


def _trace_driver_lines(trace: dict[str, Any], drl_result: dict[str, Any]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    components = trace.get("score_components", {})
    if isinstance(components, dict):
        for name, details in components.items():
            if not isinstance(details, dict):
                continue
            score = _to_float_or_none(details.get("score")) or 0.0
            ranked.append((abs(score), f"{name}: {score:+.2f}"))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = [line for _, line in ranked[:3] if line]
    if top:
        return top
    return [
        f"regime_1D: {drl_result.get('regime_1D', 'NEUTRAL')}",
        f"regime_1W: {drl_result.get('regime_1W', 'NEUTRAL')}",
    ]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _last_close(bars: list[Bar]) -> float | None:
    if not bars:
        return None
    try:
        return float(bars[-1].close)
    except Exception:
        return None


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
