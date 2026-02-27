from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import parse_iso
from app.core.replay.artifact_store import compute_policy_hash, load_artifact, save_artifact
from app.core.replay.replay_engine import replay_artifact


class DummyProvider:
    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        as_of = "2026-02-11T12:00:00-05:00"
        as_of_dt = parse_iso(as_of)

        bars: list[dict] = []
        for i in range(120):
            ts = (as_of_dt - timedelta(hours=119 - i)).isoformat()
            close = 100.0 + i * 0.2
            bars.append(
                {
                    "ts": ts,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000.0 + i * 500.0,
                }
            )

        return {"as_of": as_of, "bars": bars}


def test_artifact_save_and_load(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    now = "2026-02-11T12:00:00-05:00"
    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    context_pack = build_context_pack(
        ticker="AAPL",
        now_iso=now,
        provider=DummyProvider(),
        cache=DiskTTLCache(base_dir=str(tmp_path / "cache")),
        policy_path=str(policy_path),
        lookback_days=60,
        interval="1h",
        generate_hub_card=False,
    )

    saved_path = save_artifact(
        ticker="AAPL",
        policy_path=str(policy_path),
        context_pack=context_pack,
        now_iso=now,
        notes=["unit-test"],
    )
    assert Path(saved_path).exists()

    artifact = load_artifact(saved_path)
    assert artifact["meta"]["ticker"] == "AAPL"
    assert artifact["meta"]["policy_hash"] == compute_policy_hash(str(policy_path))
    assert artifact["meta"]["policy_id"] == "drl_v1_minimal"
    assert artifact["meta"]["policy_version"] == "1.0.0"
    assert "context_pack" in artifact
    assert "drl_result" in artifact
    assert "drl_trace" in artifact


def test_replay_determinism_same_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    now = "2026-02-11T12:00:00-05:00"
    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    context_pack = build_context_pack(
        ticker="MSFT",
        now_iso=now,
        provider=DummyProvider(),
        cache=DiskTTLCache(base_dir=str(tmp_path / "cache")),
        policy_path=str(policy_path),
        lookback_days=60,
        interval="1h",
        generate_hub_card=False,
    )

    saved_path = save_artifact(
        ticker="MSFT",
        policy_path=str(policy_path),
        context_pack=context_pack,
        now_iso=now,
        notes=["replay-test"],
    )
    artifact = load_artifact(saved_path)

    replay = replay_artifact(artifact=artifact, policy_path=str(policy_path), now_iso=now)
    assert replay["ok"] is True
    assert replay["diff"] is None


def test_replay_detects_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    now = "2026-02-11T12:00:00-05:00"
    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    context_pack = build_context_pack(
        ticker="NVDA",
        now_iso=now,
        provider=DummyProvider(),
        cache=DiskTTLCache(base_dir=str(tmp_path / "cache")),
        policy_path=str(policy_path),
        lookback_days=60,
        interval="1h",
        generate_hub_card=False,
    )

    saved_path = save_artifact(
        ticker="NVDA",
        policy_path=str(policy_path),
        context_pack=context_pack,
        now_iso=now,
        notes=["replay-mismatch"],
    )
    artifact = load_artifact(saved_path)

    artifact["drl_result"]["action_final"] = "ACCUMULATE"
    replay = replay_artifact(artifact=artifact, policy_path=str(policy_path), now_iso=now)

    assert replay["ok"] is False
    assert replay["diff"] is not None
