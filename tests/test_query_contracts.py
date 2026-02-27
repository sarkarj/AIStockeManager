from __future__ import annotations

from datetime import datetime, timezone

from app.core.marketdata.chart_fetcher import Bar, ChartSeries
from app.core.query.contracts import run_long_query, run_short_query


class _FakeQuery:
    def __init__(self) -> None:
        self.long_calls: list[tuple[str, bool]] = []
        self.pulse_calls: list[str] = []
        self.chart_calls: list[tuple[str, str]] = []
        self._quote_latest = 101.5

    def pulse_card_data(self, ticker: str) -> dict:
        self.pulse_calls.append(ticker)
        return {
            "ticker": ticker,
            "context_pack": {
                "meta": {"ticker": ticker},
                "drl": {"result": {"action_final": "WAIT", "confidence_cap": 50.0}},
                "indicators": {"as_of": "2026-02-25T10:00:00+00:00", "metrics": {"rsi_14": 47.1}},
            },
            "quote": {"latest_price": self._quote_latest, "close_price": 100.0, "prev_close_price": 99.0},
            "series_1d": _series(),
        }

    def brain_card_data(self, ticker: str, *, generate_hub_card: bool) -> dict:
        self.long_calls.append((ticker, bool(generate_hub_card)))
        payload = self.pulse_card_data(ticker=ticker)
        payload["context_pack"]["meta"]["hub_requested"] = bool(generate_hub_card)
        return payload

    def chart_series(self, ticker: str, range_key: str) -> ChartSeries:
        self.chart_calls.append((ticker, range_key))
        return _series()


def _series() -> ChartSeries:
    return ChartSeries(
        bars=[
            Bar(
                ts=datetime(2026, 2, 25, 15, 0, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.5,
                close=101.5,
                volume=1000.0,
            )
        ],
        as_of=datetime(2026, 2, 25, 15, 0, tzinfo=timezone.utc),
        source="cache",
        error=None,
        quality_flags=set(),
        cache_path=".cache/charts/test.json",
        cache_age_minutes=1.0,
        cache_hit=True,
        stale_cache=False,
        attempts=0,
    )


def test_run_short_query_contract() -> None:
    query = _FakeQuery()
    result = run_short_query("goog", market_query=query)

    assert result.ticker == "GOOG"
    assert result.drl_result["action_final"] == "WAIT"
    assert result.quote["latest_price"] == 101.5
    assert result.series_1d is not None
    assert result.series_1w is not None
    assert query.pulse_calls == ["GOOG"]
    assert ("GOOG", "1W") in query.chart_calls


def test_run_long_query_why_signature_changes_with_inputs() -> None:
    query = _FakeQuery()
    first = run_long_query("nvda", range_key="1D", include_why=True, market_query=query)
    assert first.ticker == "NVDA"
    assert first.why_signature is not None
    assert query.long_calls == [("NVDA", True)]

    query._quote_latest = 111.0
    second = run_long_query("nvda", range_key="1D", include_why=True, market_query=query)
    assert second.why_signature is not None
    assert second.why_signature != first.why_signature


def test_run_long_query_computes_signature_even_without_inline_why() -> None:
    query = _FakeQuery()
    result = run_long_query("msft", range_key="1D", include_why=False, market_query=query)
    assert result.why_signature is not None
    assert query.long_calls == [("MSFT", False)]
