from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.marketdata.chart_fetcher import Bar, ChartSeries
from app.core.orchestration.time_utils import now_iso, parse_iso
from app.ui.components import brain
from app.ui.components.brain_market_card import GAUGE_TOOLTIPS, _normalize_refs
from app.ui.viewmodels.brain_market_vm import BrainMarketVM, Gauge, QuoteStats
from app.ui.viewmodels.brain_vm import build_brain_view_model
from app.ui.viewmodels.brain_vm import compute_display_price


def _context_pack_without_hub() -> dict:
    return {
        "meta": {
            "ticker": "AAPL",
            "generated_at": "2026-02-11T12:00:00-05:00",
            "data_quality": {
                "notes": ["BEDROCK_UNAVAILABLE: hub card fallback skipped"],
                "prices": {"age_minutes": 5.0, "stale": False},
                "overall_stale": False,
            },
        },
        "prices": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "bars": [
                {
                    "ts": "2026-02-11T11:00:00-05:00",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000.0,
                },
                {
                    "ts": "2026-02-11T12:00:00-05:00",
                    "open": 100.5,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "volume": 1200.0,
                },
            ],
        },
        "indicators": {
            "as_of": "2026-02-11T12:00:00-05:00",
            "metrics": {
                "price_last": 101.5,
                "ema_50": 99.5,
                "sma_200": 95.0,
                "rsi_14": 55.0,
                "macd": 0.8,
                "macd_signal": 0.6,
                "stoch_k": 48.0,
                "adx_14": 23.0,
                "vroc_14": 12.0,
                "atr_pct": 2.2,
            },
        },
        "drl": {
            "result": {
                "regime_1D": "BULL",
                "regime_1W": "BULL",
                "action_final": "ACCUMULATE",
                "confidence_cap": 75,
                "gates_triggered": [],
                "conflicts": [],
                "decision_trace": {
                    "score_final": 8.2,
                    "score_raw": 6.7,
                    "score_components": {
                        "regime_score": {"score": 4.0},
                        "momentum_score": {"score": 3.0},
                        "participation_score": {"score": 1.0},
                    },
                },
            }
        },
    }


def _bars(start, n: int, step: timedelta, base: float = 100.0) -> list[dict]:
    values: list[dict] = []
    for i in range(n):
        ts = (start + step * i).isoformat()
        close = base + (i * 0.7)
        values.append(
            {
                "ts": ts,
                "open": close - 0.3,
                "high": close + 0.5,
                "low": close - 0.6,
                "close": close,
                "volume": 1000.0 + i,
            }
        )
    return values


def test_brain_fallback_contains_deterministic_reason() -> None:
    context_pack = _context_pack_without_hub()
    vm = build_brain_view_model(context_pack)

    fallback = brain.build_why_fallback(context_pack=context_pack, vm=vm)

    assert "Deterministic fallback" in fallback["title"]
    assert fallback["reason"] in {
        "LLM not configured (run make llm-smoke; ensure .env loaded).",
        "LLM reachable but hub generation disabled",
        "Hub validation failed",
        "Context pack missing hub artifact",
    }
    assert "AI Signal" not in brain.brain_sections()


@pytest.mark.parametrize(
    ("range_key", "advanced_source", "expected_period", "expected_interval", "expected_prepost"),
    [
        ("1D", None, "1d", "5m", True),
        ("1W", None, "5d", "30m", True),
        ("1M", None, "1mo", "1h", True),
        ("3M", None, "3mo", "1d", False),
        ("YTD", None, "ytd", "1d", False),
        ("1Y", None, "1y", "1d", False),
        ("Advanced", "1M", "1mo", "1h", True),
    ],
)
def test_range_contract_provider_call(
    monkeypatch,
    range_key: str,
    advanced_source: str | None,
    expected_period: str,
    expected_interval: str,
    expected_prepost: bool,
) -> None:
    calls: list[dict] = []

    class FakeFetcher:
        def fetch_chart_series(self, ticker: str, range_key: str) -> ChartSeries:
            calls.append({"ticker": ticker, "range_key": range_key})
            now = parse_iso(now_iso())
            bars = [
                Bar(
                    ts=now - timedelta(hours=2),
                    open=200.0,
                    high=201.0,
                    low=199.8,
                    close=200.7,
                    volume=1000.0,
                ),
                Bar(
                    ts=now - timedelta(hours=1),
                    open=200.8,
                    high=202.0,
                    low=200.5,
                    close=201.5,
                    volume=1200.0,
                ),
            ]
            return ChartSeries(
                bars=bars,
                as_of=bars[-1].ts,
                source="live",
                error=None,
                quality_flags=set(),
                cache_path=str("cache_path"),
                cache_age_minutes=0.0,
                cache_hit=False,
                stale_cache=False,
                attempts=1,
            )

    monkeypatch.setattr(brain, "_get_chart_fetcher", lambda: FakeFetcher())
    series = brain._get_price_series_for_range(
        ticker="AAPL",
        range_key=range_key,
        advanced_source_range=advanced_source,
        price_hint=200.0,
    )

    assert calls, "expected provider call"
    assert calls[0]["range_key"] == ("1M" if range_key == "Advanced" and advanced_source == "1M" else range_key)
    assert series["contract"]["period"] == expected_period
    assert series["contract"]["interval"] == expected_interval
    assert series["contract"]["prepost"] is expected_prepost
    assert series["synthetic"] is False


def test_chart_builder_uses_input_series_without_wave() -> None:
    now = parse_iso(now_iso())
    raw_bars = _bars(now - timedelta(days=20), 50, timedelta(days=1), base=50.0)
    raw_bars = list(reversed(raw_bars))

    series = brain.build_chart_series_from_bars(
        range_key="1M",
        bars=raw_bars,
        as_of=raw_bars[0]["ts"],
        source="unit_test",
        now_iso_value=now_iso(),
        min_points=10,
        max_points=512,
    )

    closes = [float(b["close"]) for b in series["bars"]]
    assert closes == sorted(closes)
    assert series["synthetic"] is False


def test_1d_point_bounds_snapshot() -> None:
    now = parse_iso(now_iso()).astimezone()
    start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    bars = _bars(start, 240, timedelta(minutes=5), base=300.0)

    series = brain.build_chart_series_from_bars(
        range_key="1D",
        bars=bars,
        as_of=bars[-1]["ts"],
        source="unit_test",
        now_iso_value=now_iso(),
        min_points=6,
        max_points=128,
    )

    assert 20 <= series["point_count"] <= 128


def test_no_flat_fallback_label_when_no_data(monkeypatch) -> None:
    class EmptyFetcher:
        def fetch_chart_series(self, ticker: str, range_key: str) -> ChartSeries:
            now = parse_iso(now_iso())
            return ChartSeries(
                bars=[],
                as_of=now,
                source="none",
                error="empty_live",
                quality_flags={"EMPTY_LIVE"},
                cache_path="none",
                cache_age_minutes=None,
                cache_hit=False,
                stale_cache=False,
                attempts=3,
            )

    monkeypatch.setattr(brain, "_get_chart_fetcher", lambda: EmptyFetcher())
    series = brain._get_price_series_for_range(ticker="GOOG", range_key="1D", price_hint=100.0)
    assert series["source"] == "none"
    assert "FLAT_FALLBACK" not in series.get("flags", [])


def test_hub_narrative_allows_stale_cache_flag_only() -> None:
    context_pack = {
        "meta": {
            "data_quality": {
                "overall_stale": False,
            }
        }
    }
    vm = {
        "last_price": 101.25,
        "quote": {"source": "cache"},
    }
    primary_series = {"source": "cache", "flags": ["STALE_CACHE"]}
    assert brain._should_render_hub_narrative(context_pack=context_pack, vm=vm, primary_series=primary_series) is True


def test_attach_why_meta_preserves_loaded_signature() -> None:
    context_pack = {"meta": {"hub": {"why_signature": "loaded-signature"}}}
    brain._attach_why_meta(
        context_pack=context_pack,
        why_signature="requested-signature",
        why_cache_state="cache_fallback",
    )
    hub = context_pack["meta"]["hub"]
    assert hub["why_signature"] == "loaded-signature"
    assert hub["why_signature_requested"] == "requested-signature"
    assert hub["why_signature_loaded"] == "loaded-signature"


def test_canonical_price_prefers_bars_and_flags_mismatch() -> None:
    result = compute_display_price(
        indicators={"price_last": 100.0},
        series_for_selected_range={
            "bars": [
                {"ts": "2026-02-11T11:00:00-05:00", "close": 103.0},
                {"ts": "2026-02-11T12:00:00-05:00", "close": 102.5},
            ]
        },
    )
    assert result["display_price"] == 102.5
    assert result["price_source"] == "bars_close_last"
    assert "PRICE_MISMATCH_GT_1PCT" in result["price_sanity_flags"]


def test_timing_present_for_cache_hit_series() -> None:
    now = parse_iso(now_iso())
    vm = BrainMarketVM(
        ticker="AAPL",
        price_display=100.0,
        as_of_price=now,
        chart_series=ChartSeries(
            bars=[
                Bar(ts=now - timedelta(hours=1), open=99.0, high=101.0, low=98.0, close=100.0, volume=1000.0),
                Bar(ts=now, open=100.0, high=102.0, low=99.0, close=101.0, volume=1200.0),
            ],
            as_of=now,
            source="cache",
            error=None,
            quality_flags=set(),
            cache_path=".cache/charts/example.json",
            cache_age_minutes=1.0,
            cache_hit=True,
            stale_cache=False,
            attempts=0,
        ),
        quote=QuoteStats(
            open=99.0,
            volume=2200,
            day_low=98.0,
            day_high=102.0,
            year_low=80.0,
            year_high=130.0,
            source_range_day="1D",
            source_range_year="1Y",
            notes=[],
        ),
        gauges=[Gauge(label="TREND", value="BULLISH", score=80.0, tone="good")] * 4,
        badge_state={"grounded": True, "degraded": False, "tool_down": False, "stale": False},
        why_block={"mode": "FALLBACK"},
        chart_source_range="1D",
        gauge_basis_note="Gauges based on 1D indicators",
    )
    series_dict = brain._market_vm_series_to_dict(vm=vm, range_key="1D", fetch_ms=12.34)
    assert isinstance(series_dict["diagnostics"]["fetch_ms"], float)
    assert series_dict["diagnostics"]["fetch_ms"] == 0.0
    assert series_dict["diagnostics"]["cache_hit"] is True


def test_citation_dedupe_and_prefix_cleanup() -> None:
    refs = _normalize_refs(
        [
            "Refs: indicator:rsi_14",
            "Citations: indicator:rsi_14",
            "indicator:macd",
            "indicator:macd",
        ]
    )
    assert refs == ["rsi_14", "macd"]


def test_strength_gauge_tooltip_contract_text() -> None:
    assert "EMA50 vs SMA200 vs price structure" in GAUGE_TOOLTIPS["TREND"]
    assert "RSI + MACD (+ signal) + Stoch" in GAUGE_TOOLTIPS["MOMENTUM"]
    assert "ATR%" in GAUGE_TOOLTIPS["RISK"]
    assert "ADX (trend strength)" in GAUGE_TOOLTIPS["STRENGTH"]
    assert "VROC (volume participation)" in GAUGE_TOOLTIPS["STRENGTH"]


def test_attach_why_meta_carries_sync_and_usage_fields() -> None:
    context_pack = {"meta": {"hub": {"status": "missing"}}}
    brain._attach_why_meta(
        context_pack=context_pack,
        why_signature="abc123",
        why_cache_state="live_sync",
        why_sync_status="success",
        why_sync_error="",
        why_sync_elapsed_ms=123.45,
        why_llm_usage={"input_tokens": 11, "output_tokens": 22, "total_tokens": 33, "latency_ms": 456.7},
    )
    hub = context_pack["meta"]["hub"]
    assert hub["why_signature_requested"] == "abc123"
    assert hub["why_source"] == "live_sync"
    assert hub["why_sync_status"] == "success"
    assert float(hub["why_sync_elapsed_ms"]) == 123.45
    assert hub["llm_usage"]["total_tokens"] == 33
