from __future__ import annotations

from pathlib import Path

from app.core.portfolio.portfolio_store import load_portfolio, remove_holding, save_portfolio, upsert_holding


def test_portfolio_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.json"

    portfolio = load_portfolio(path=str(path))
    assert portfolio.holdings == []

    portfolio = upsert_holding(ticker="aapl", avg_cost=100.0, quantity=2.0, path=str(path))
    assert len(portfolio.holdings) == 1
    assert portfolio.holdings[0].ticker == "AAPL"

    save_portfolio(portfolio, path=str(path))
    reloaded = load_portfolio(path=str(path))
    assert len(reloaded.holdings) == 1
    assert reloaded.holdings[0].avg_cost == 100.0
    assert reloaded.holdings[0].quantity == 2.0

    updated = upsert_holding(ticker="AAPL", avg_cost=110.0, quantity=3.0, path=str(path))
    assert len(updated.holdings) == 1
    assert updated.holdings[0].avg_cost == 110.0
    assert updated.holdings[0].quantity == 3.0

    removed = remove_holding("AAPL", path=str(path))
    assert removed.holdings == []

    reloaded_after_remove = load_portfolio(path=str(path))
    assert reloaded_after_remove.holdings == []
