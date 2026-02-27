from __future__ import annotations

from app.core.context_pack import why_cache


def test_build_why_signature_is_deterministic_and_changes_with_inputs() -> None:
    base = {
        "ticker": "GOOG",
        "drl_result": {"action_final": "WAIT", "confidence_cap": 50, "gates_triggered": [], "conflicts": []},
        "indicators": {"as_of": "2026-02-25T10:00:00+00:00", "metrics": {"rsi_14": 47.1}},
        "quote": {"latest_price": 100.0, "close_price": 99.0, "after_hours_price": 100.0, "prev_close_price": 98.0},
        "range_key": "1D",
    }
    first = why_cache.build_why_signature(**base)
    second = why_cache.build_why_signature(**base)
    assert first == second

    changed = dict(base)
    changed["quote"] = dict(base["quote"])
    changed["quote"]["latest_price"] = 103.0
    third = why_cache.build_why_signature(**changed)
    assert third != first


def test_save_load_and_hydrate_why_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(why_cache, "WHY_CACHE_DIR", tmp_path / "why")
    signature = "abc123" * 10 + "zz"
    hub_card = {"summary": {"one_liner": "deterministic"}, "meta": {"mode": "NORMAL"}}
    hub_meta = {"status": "present", "mode": "NORMAL", "hub_valid": True}
    saved = why_cache.save_why_artifact(
        signature=signature,
        ticker="NVDA",
        hub_card=hub_card,
        hub_meta=hub_meta,
        generated_at="2026-02-25T10:00:00+00:00",
    )
    assert saved

    loaded = why_cache.load_why_artifact(signature=signature, ticker="NVDA")
    assert loaded is not None
    assert loaded.signature == signature
    assert loaded.hub_card["summary"]["one_liner"] == "deterministic"

    context_pack = {"meta": {"ticker": "NVDA", "hub": {"status": "missing"}}, "drl": {"result": {}}}
    hydrated = why_cache.hydrate_context_pack_with_why(context_pack, loaded)
    assert hydrated["hub_card"]["summary"]["one_liner"] == "deterministic"
    assert hydrated["meta"]["hub"]["status"] == "present"
    assert hydrated["meta"]["hub"]["why_signature"] == signature


def test_load_latest_why_artifact_prefers_newest_and_respects_age(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(why_cache, "WHY_CACHE_DIR", tmp_path / "why")
    first_sig = "a" * 64
    second_sig = "b" * 64
    hub_card = {"summary": {"one_liner": "deterministic"}, "meta": {"mode": "NORMAL"}}

    why_cache.save_why_artifact(
        signature=first_sig,
        ticker="GOOG",
        hub_card=hub_card,
        hub_meta={"status": "present", "mode": "NORMAL", "hub_valid": True},
        generated_at="2026-02-25T10:00:00+00:00",
    )
    why_cache.save_why_artifact(
        signature=second_sig,
        ticker="GOOG",
        hub_card=hub_card,
        hub_meta={"status": "present", "mode": "NORMAL", "hub_valid": True},
        generated_at="2026-02-25T10:10:00+00:00",
    )

    latest = why_cache.load_latest_why_artifact(ticker="GOOG")
    assert latest is not None
    assert latest.signature == second_sig

    too_fresh = why_cache.load_latest_why_artifact(ticker="GOOG", max_age_minutes=1)
    assert too_fresh is None
