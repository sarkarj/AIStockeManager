from __future__ import annotations

from app.core.marketdata.price_sanity import reconcile_price_last


def test_primary_close_wins_for_display() -> None:
    result = reconcile_price_last(
        ticker="AAPL",
        indicator_price_last=192.1,
        primary_series_close=191.8,
        fallback_series_close=190.0,
    )

    assert result.display_price == 191.8
    assert result.source == "bars"
    assert "MISSING_BARS" not in result.quality_flags


def test_fallback_close_used_when_primary_missing() -> None:
    result = reconcile_price_last(
        ticker="MSFT",
        indicator_price_last=410.0,
        primary_series_close=None,
        fallback_series_close=405.5,
    )

    assert result.display_price == 405.5
    assert result.source == "bars"
    assert "MISSING_BARS" in result.quality_flags


def test_missing_primary_and_fallback_returns_none() -> None:
    result = reconcile_price_last(
        ticker="PLUG",
        indicator_price_last=203.72,
        primary_series_close=None,
        fallback_series_close=None,
    )

    assert result.display_price is None
    assert result.source == "none"
    assert "MISSING_BARS" in result.quality_flags


def test_indicator_mismatch_flagged_and_indicator_not_used() -> None:
    result = reconcile_price_last(
        ticker="PLUG",
        indicator_price_last=203.72,
        primary_series_close=1.89,
        fallback_series_close=None,
    )

    assert result.display_price == 1.89
    assert result.source == "bars"
    assert "PRICE_MISMATCH" in result.quality_flags
    assert result.note is not None
