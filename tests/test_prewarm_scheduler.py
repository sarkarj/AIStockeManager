from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.core.marketdata import prewarm


class FakeQuery:
    def revalidate_tickers(self, *, tickers: set[str], range_keys: tuple[str, ...]) -> dict[str, int]:
        return {
            "attempted": len(tickers) * len(range_keys),
            "live": len(tickers),
            "cache": 0,
            "none": 0,
            "errors": 0,
        }


def _use_tmp_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prewarm, "PREWARM_DIR", tmp_path / "prewarm")
    monkeypatch.setattr(prewarm, "PREWARM_CONFIG_PATH", (tmp_path / "prewarm" / "config.json"))
    monkeypatch.setattr(prewarm, "PREWARM_QUEUE_PATH", (tmp_path / "prewarm" / "queue.json"))
    monkeypatch.setattr(prewarm, "PREWARM_STATUS_PATH", (tmp_path / "prewarm" / "status.json"))
    monkeypatch.setattr(prewarm, "PREWARM_LOCK_PATH", (tmp_path / "prewarm" / "worker.lock"))


def test_queue_roundtrip(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    request = prewarm.enqueue_prewarm_request(
        scope="visible",
        tickers={"GOOG", "NVDA"},
        range_keys=("1D", "1W"),
        reason="manual_refresh",
    )
    assert request["scope"] == "visible"
    popped = prewarm.pop_prewarm_requests(max_items=1)
    assert len(popped) == 1
    assert set(popped[0]["tickers"]) == {"GOOG", "NVDA"}
    assert prewarm.pop_prewarm_requests(max_items=1) == []


def test_queue_depth(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    assert prewarm.prewarm_queue_depth() == 0
    prewarm.enqueue_prewarm_request(scope="visible", tickers={"AAPL"}, range_keys=("1D",), reason="manual_refresh")
    assert prewarm.prewarm_queue_depth() == 1
    _ = prewarm.pop_prewarm_requests(max_items=1)
    assert prewarm.prewarm_queue_depth() == 0


def test_schedule_due_respects_interval(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    cfg = prewarm.load_prewarm_config()
    now_et = datetime(2026, 2, 25, 10, 0, tzinfo=prewarm._ET)
    due, bucket, _ = prewarm.scheduled_run_due(config=cfg, status={}, now_et=now_et)
    assert due is True
    assert bucket == "market"

    status = {"last_scheduled_at": (now_et - timedelta(minutes=10)).isoformat()}
    due2, _, _ = prewarm.scheduled_run_due(config=cfg, status=status, now_et=now_et)
    assert due2 is False

    status2 = {"last_scheduled_at": (now_et - timedelta(minutes=31)).isoformat()}
    due3, _, _ = prewarm.scheduled_run_due(config=cfg, status=status2, now_et=now_et)
    assert due3 is True


def test_holiday_bucket_uses_weekend_cadence(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    holiday = datetime(2026, 12, 25, 12, 0, tzinfo=prewarm._ET)
    bucket = prewarm.resolve_schedule_bucket(holiday)
    assert bucket == "weekend"


def test_run_revalidate_job_uses_query_contract(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    result = prewarm.run_revalidate_job(
        query=FakeQuery(),
        scope="brain",
        tickers={"AAPL", "MSFT"},
        range_keys=("1D", "1W", "1M"),
        reason="manual_refresh",
    )
    assert result["attempted"] == 6
    assert result["live"] == 2
    assert result["errors"] == 0


def test_should_revalidate_uses_bucket_cadence() -> None:
    now_et = datetime(2026, 2, 25, 11, 0, tzinfo=prewarm._ET)
    policy = {"market": 30, "after_hours": 60, "off_hours": 720, "weekend": 720}
    assert (
        prewarm.should_revalidate(
            ticker="AAPL",
            range_key="1D",
            last_updated_at=now_et - timedelta(minutes=45),
            now_et=now_et,
            market_bucket="market",
            policy_minutes=policy,
        )
        is True
    )
    assert (
        prewarm.should_revalidate(
            ticker="AAPL",
            range_key="1D",
            last_updated_at=now_et - timedelta(minutes=10),
            now_et=now_et,
            market_bucket="market",
            policy_minutes=policy,
        )
        is False
    )


def test_enqueue_refresh_request_roundtrip(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    request_id = prewarm.enqueue_refresh_request(
        ticker="MSFT",
        range_keys=("1D", "1W"),
        reason="ticker_click",
        priority=1,
    )
    assert request_id
    items = prewarm.pop_prewarm_requests(max_items=1)
    assert len(items) == 1
    assert items[0]["reason"] == "ticker_click"
    assert items[0]["tickers"] == ["MSFT"]


def test_process_refresh_batch_handles_why_refresh(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    prewarm.enqueue_prewarm_request(
        scope="brain",
        tickers={"GOOG"},
        range_keys=("1D",),
        reason="why_refresh",
        metadata={"why_signature": "sig-123"},
    )

    def _fake_why_refresh(*, ticker: str, range_key: str = "1D", expected_signature: str | None = None):
        assert ticker == "GOOG"
        assert range_key == "1D"
        assert expected_signature == "sig-123"
        return {"attempted": 1, "live": 1, "cache": 0, "none": 0, "errors": 0}

    monkeypatch.setattr(prewarm, "run_why_refresh_job", _fake_why_refresh)
    stats = prewarm.process_refresh_batch(max_items=1, now_et=datetime(2026, 2, 25, 11, 0, tzinfo=prewarm._ET))
    assert stats["processed"] == 1
    assert stats["attempted"] == 1
    assert stats["live"] == 1


def test_run_why_refresh_job_preserves_expected_signature(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)

    saved_signatures: list[str] = []

    def _fake_load_context_pack_for_query(**kwargs):
        return {
            "drl": {"result": {"action_final": "WAIT", "confidence_cap": 50, "gates_triggered": [], "conflicts": []}},
            "indicators": {"as_of": "2026-02-25T10:00:00+00:00", "metrics": {"rsi_14": 47.1}},
            "hub_card": {"summary": {"one_liner": "deterministic"}, "meta": {"mode": "NORMAL"}},
            "meta": {"hub": {"status": "present", "mode": "NORMAL", "hub_valid": True}},
        }

    monkeypatch.setattr(
        "app.core.query.context_loader.load_context_pack_for_query",
        _fake_load_context_pack_for_query,
    )
    monkeypatch.setattr("app.core.marketdata.quotes.get_quote_snapshot", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.core.marketdata.quotes.quote_snapshot_to_dict",
        lambda *_args, **_kwargs: {
            "latest_price": 100.0,
            "close_price": 99.0,
            "after_hours_price": 100.1,
            "prev_close_price": 98.5,
            "latest_ts": "2026-02-25T10:00:00+00:00",
            "close_ts": "2026-02-24T16:00:00+00:00",
            "after_hours_ts": "2026-02-24T19:55:00+00:00",
        },
    )
    monkeypatch.setattr("app.core.context_pack.why_cache.build_why_signature", lambda **kwargs: "resolved-signature")

    def _fake_save_why_artifact(*, signature: str, **kwargs):
        saved_signatures.append(signature)
        return str(tmp_path / f"{signature}.json")

    monkeypatch.setattr("app.core.context_pack.why_cache.save_why_artifact", _fake_save_why_artifact)

    result = prewarm.run_why_refresh_job(
        ticker="GOOG",
        range_key="1D",
        expected_signature="expected-signature",
    )
    assert result["why_status"] == "saved"
    assert result["why_signature"] == "expected-signature"
    assert "expected-signature" in saved_signatures
    assert "resolved-signature" in saved_signatures


def test_cache_hygiene_snapshot(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "a.json").write_text("{}", encoding="utf-8")
    (charts_dir / "b.json").write_text("", encoding="utf-8")
    snapshot = prewarm.cache_hygiene_snapshot(cache_dirs=(charts_dir,))
    assert snapshot["file_count"] == 2
    assert snapshot["empty_files"] == 1
    assert snapshot["total_size_mb"] >= 0.0


def test_load_prewarm_config_enforces_full_brain_ranges(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    config = prewarm.load_prewarm_config_from_payload({"main_range_keys": ["1D", "1W"]})
    assert config["main_range_keys"] == list(prewarm.BRAIN_RANGE_KEYS)


def test_save_prewarm_config_migrates_existing_main_ranges(monkeypatch, tmp_path: Path) -> None:
    _use_tmp_paths(monkeypatch, tmp_path)
    saved = prewarm.save_prewarm_config({"main_range_keys": ["1D", "1W", "1M"]})
    assert saved["main_range_keys"] == list(prewarm.BRAIN_RANGE_KEYS)
