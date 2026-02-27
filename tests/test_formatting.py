from __future__ import annotations

from app.ui.utils.formatting import format_int, format_money


def test_format_money_none() -> None:
    assert format_money(None) == "—"


def test_format_money_numeric() -> None:
    assert format_money(1234.5) == "$1,234.50"


def test_format_money_nan() -> None:
    assert format_money(float("nan")) == "—"


def test_format_int_numeric() -> None:
    assert format_int(16669105) == "16,669,105"
