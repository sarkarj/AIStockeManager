from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import parse_iso


class DummyProvider:
    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        as_of = "2026-02-11T12:00:00-05:00"
        as_of_dt = parse_iso(as_of)

        bars: list[dict] = []
        for i in range(10):
            ts = (as_of_dt - timedelta(hours=9 - i)).isoformat()
            close = 100.0 + i
            bars.append(
                {
                    "ts": ts,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000.0 + i * 1_000.0,
                }
            )

        return {"as_of": as_of, "bars": bars}


def test_context_pack_schema(tmp_path: Path) -> None:
    now = "2026-02-11T12:00:00-05:00"
    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    pack = build_context_pack(
        ticker="AAPL",
        now_iso=now,
        provider=DummyProvider(),
        cache=DiskTTLCache(base_dir=str(tmp_path / "cache")),
        policy_path=str(policy_path),
        lookback_days=60,
        interval="1h",
    )

    assert set(pack.keys()) == {"meta", "prices", "indicators", "drl"}

    drl_result = pack["drl"]["result"]
    assert "action_final" in drl_result
    assert "confidence_cap" in drl_result

    prices_stale = pack["meta"]["data_quality"]["prices"]["stale"]
    assert isinstance(prices_stale, bool)

    bars = pack["prices"]["bars"]
    assert isinstance(bars, list)
    assert bars

    first = bars[0]
    for field in ["ts", "open", "high", "low", "close", "volume"]:
        assert field in first
