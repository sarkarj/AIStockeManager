from __future__ import annotations

from app.ui.components.ui_utils import (
    action_to_pill_class,
    action_to_ui_label,
    format_money,
    format_pct,
)


def test_action_to_ui_label_mappings() -> None:
    assert action_to_ui_label("ACCUMULATE") == "BUY"
    assert action_to_ui_label("WAIT") == "HOLD"
    assert action_to_ui_label("REDUCE") == "SELL"
    assert action_to_ui_label("UNKNOWN") == "HOLD"


def test_action_to_pill_class_mappings() -> None:
    assert action_to_pill_class("ACCUMULATE") == "pill-buy"
    assert action_to_pill_class("WAIT") == "pill-hold"
    assert action_to_pill_class("REDUCE") == "pill-sell"
    assert action_to_pill_class("UNKNOWN") == "pill-hold"


def test_format_helpers_do_not_crash() -> None:
    assert "$" in format_money(123.45)
    assert "%" in format_pct(1.23)
    assert "$" in format_money("x")
    assert "%" in format_pct("x")
