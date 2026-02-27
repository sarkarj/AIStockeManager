from __future__ import annotations

from datetime import datetime, timezone

from app.core.marketdata.quotes import QuoteSnapshot, compute_quote_display, quote_snapshot_to_dict


def test_quote_math_closed_market_prefers_after_hours_and_uses_close_for_today() -> None:
    q = QuoteSnapshot(
        ticker="GOOG",
        as_of=datetime(2026, 2, 15, 1, 0, tzinfo=timezone.utc),
        prev_close=300.0,
        close=306.0,
        close_ts=datetime(2026, 2, 14, 21, 0, tzinfo=timezone.utc),
        after_hours=305.5,
        after_hours_ts=datetime(2026, 2, 14, 22, 30, tzinfo=timezone.utc),
        last_regular=306.0,
        last_regular_ts=datetime(2026, 2, 14, 21, 0, tzinfo=timezone.utc),
    )

    display = compute_quote_display(q, market_state="closed")
    assert display.latest_price == 305.5
    assert display.latest_label == "after_hours"
    assert display.today_abs == 6.0
    assert display.today_pct == 0.02
    assert display.after_hours_abs == -0.5
    assert round(float(display.after_hours_pct or 0.0), 6) == round(-0.5 / 306.0, 6)


def test_quote_math_missing_after_hours_uses_regular() -> None:
    q = QuoteSnapshot(
        ticker="AAPL",
        as_of=datetime(2026, 2, 15, 1, 0, tzinfo=timezone.utc),
        prev_close=184.0,
        close=185.0,
        close_ts=datetime(2026, 2, 14, 21, 0, tzinfo=timezone.utc),
        after_hours=None,
        after_hours_ts=None,
        last_regular=185.2,
        last_regular_ts=datetime(2026, 2, 14, 20, 58, tzinfo=timezone.utc),
    )

    display = compute_quote_display(q, market_state="closed")
    assert display.latest_price == 185.2
    assert display.latest_label == "regular"
    assert display.today_abs == 1.0
    assert display.today_pct == (1.0 / 184.0)
    assert display.after_hours_abs is None
    assert display.after_hours_pct is None


def test_quote_math_never_returns_strings_or_nans() -> None:
    q = QuoteSnapshot(
        ticker="MSFT",
        as_of=datetime(2026, 2, 15, 1, 0, tzinfo=timezone.utc),
        prev_close=None,
        close=None,
        close_ts=None,
        after_hours=None,
        after_hours_ts=None,
        last_regular=None,
        last_regular_ts=None,
    )
    display = compute_quote_display(q, market_state="unknown")
    assert display.latest_price is None
    assert display.today_abs is None
    assert display.today_pct is None
    assert display.after_hours_abs is None
    assert display.after_hours_pct is None


def test_quote_snapshot_hides_extended_session_during_regular_hours() -> None:
    q = QuoteSnapshot(
        ticker="GOOG",
        as_of=datetime(2026, 2, 25, 19, 0, tzinfo=timezone.utc),  # 14:00 ET
        prev_close=300.0,
        close=306.0,
        close_ts=datetime(2026, 2, 24, 21, 0, tzinfo=timezone.utc),
        after_hours=312.0,
        after_hours_ts=datetime(2026, 2, 25, 14, 25, tzinfo=timezone.utc),  # 09:25 ET pre-market
        last_regular=310.7,
        last_regular_ts=datetime(2026, 2, 25, 18, 55, tzinfo=timezone.utc),
    )
    payload = quote_snapshot_to_dict(q, market_state="open")
    assert payload["session_state"] == "REGULAR"
    assert payload["show_extended_session"] is False
    assert payload["extended_label"] is None
