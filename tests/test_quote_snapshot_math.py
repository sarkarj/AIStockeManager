from __future__ import annotations

from datetime import datetime, timezone

from app.core.marketdata.quotes import QuoteSnapshot, compute_quote_display


def test_quote_snapshot_math_values() -> None:
    q = QuoteSnapshot(
        ticker="GOOG",
        as_of=datetime(2026, 2, 15, 1, 0, tzinfo=timezone.utc),
        prev_close=309.37,
        close=306.02,
        close_ts=datetime(2026, 2, 14, 21, 0, tzinfo=timezone.utc),
        after_hours=305.81,
        after_hours_ts=datetime(2026, 2, 14, 22, 30, tzinfo=timezone.utc),
        last_regular=306.02,
        last_regular_ts=datetime(2026, 2, 14, 21, 0, tzinfo=timezone.utc),
    )
    d = compute_quote_display(q, market_state="closed")
    assert d.latest_price == 305.81
    assert d.today_abs == (306.02 - 309.37)
    assert d.after_hours_abs == (305.81 - 306.02)
