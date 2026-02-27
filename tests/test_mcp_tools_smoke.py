from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from app.core.mcp import server
from app.core.mcp.tools import events, macro, news, prices
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import parse_iso


class DummyMarketDataProvider:
    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        as_of = "2026-02-11T12:00:00-05:00"
        as_of_dt = parse_iso(as_of)

        bars: list[dict] = []
        if interval.endswith("d"):
            step = timedelta(days=1)
            count = max(3, lookback_days + 1)
        else:
            step = timedelta(hours=1)
            count = max(24, lookback_days * 2)

        for i in range(count):
            ts = (as_of_dt - step * (count - 1 - i)).isoformat()
            base = 100.0 + i
            bars.append(
                {
                    "ts": ts,
                    "open": base - 0.3,
                    "high": base + 0.8,
                    "low": base - 0.9,
                    "close": base,
                    "volume": 1_000_000.0 + i * 1000.0,
                }
            )

        return {"as_of": as_of, "bars": bars}


def test_mcp_tool_registration() -> None:
    tool_defs = server.get_tool_definitions()
    names = {td["name"] for td in tool_defs}

    assert names == {
        "stock.get_prices",
        "stock.get_news",
        "stock.get_macro_snapshot",
        "stock.get_events",
    }

    for td in tool_defs:
        assert "input_schema" in td
        assert td["input_schema"]["type"] == "object"


def test_mcp_tool_handlers_smoke(tmp_path: Path) -> None:
    cache = DiskTTLCache(base_dir=str(tmp_path / "cache"))
    provider = DummyMarketDataProvider()
    now = "2026-02-11T12:00:00-05:00"

    prices_result = prices.get_prices(
        ticker="AAPL",
        interval="1h",
        lookback_days=10,
        provider=provider,
        cache=cache,
        now_iso=now,
    )
    assert {"ticker", "as_of", "bars", "source"}.issubset(prices_result.keys())

    news_result = news.get_news(
        ticker="AAPL",
        lookback_hours=24,
        max_items=5,
        cache=cache,
        now_iso=now,
    )
    assert {"ticker", "as_of", "items", "source"}.issubset(news_result.keys())
    assert isinstance(news_result["items"], list)

    macro_result = macro.get_macro_snapshot(
        lookback_days=1,
        provider=provider,
        cache=cache,
        now_iso=now,
    )
    assert {"as_of", "items", "market_mode", "source"}.issubset(macro_result.keys())

    events_result = events.get_events(
        ticker="AAPL",
        max_items=10,
        cache=cache,
        now_iso=now,
    )
    assert {"ticker", "as_of", "items", "source"}.issubset(events_result.keys())
    assert isinstance(events_result["items"], list)
