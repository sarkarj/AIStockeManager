from __future__ import annotations

import errno
import json
import os
from pathlib import Path

from app.core.portfolio.portfolio_schema import Holding, Portfolio

DEFAULT_PORTFOLIO_PATH = "app/core/portfolio/portfolio.json"


def load_portfolio(path: str = DEFAULT_PORTFOLIO_PATH) -> Portfolio:
    path_obj = Path(path)
    _ensure_portfolio_file(path_obj)

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {"holdings": []}

    portfolio = Portfolio.model_validate(data)
    deduped = _dedupe_holdings(portfolio.holdings)
    if len(deduped) != len(portfolio.holdings):
        portfolio = Portfolio(holdings=deduped)
        save_portfolio(portfolio, path=path)
    return portfolio


def save_portfolio(portfolio: Portfolio, path: str = DEFAULT_PORTFOLIO_PATH) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    deduped = _dedupe_holdings(portfolio.holdings)
    payload = Portfolio(holdings=deduped).model_dump(mode="json", exclude_none=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")

    try:
        os.replace(tmp_path, path_obj)
    except OSError as exc:
        # Some Docker bind-mounted file targets cannot be atomically replaced.
        if exc.errno not in {errno.EBUSY, errno.EXDEV, errno.EPERM}:
            raise
        with tmp_path.open("r", encoding="utf-8") as src, path_obj.open("w", encoding="utf-8") as dst:
            dst.write(src.read())
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def upsert_holding(
    ticker: str,
    avg_cost: float,
    quantity: float,
    tags: list[str] | None = None,
    path: str = DEFAULT_PORTFOLIO_PATH,
) -> Portfolio:
    portfolio = load_portfolio(path=path)
    ticker_norm = ticker.strip().upper()

    deduped = _dedupe_holdings(portfolio.holdings)
    updated = False
    holdings: list[Holding] = []
    for holding in deduped:
        if holding.ticker == ticker_norm:
            holdings.append(Holding(ticker=ticker_norm, avg_cost=avg_cost, quantity=quantity, tags=tags))
            updated = True
        else:
            holdings.append(holding)

    if not updated:
        holdings.append(Holding(ticker=ticker_norm, avg_cost=avg_cost, quantity=quantity, tags=tags))

    holdings = sorted(_dedupe_holdings(holdings), key=lambda h: h.ticker)
    new_portfolio = Portfolio(holdings=holdings)
    save_portfolio(new_portfolio, path=path)
    return new_portfolio


def remove_holding(ticker: str, path: str = DEFAULT_PORTFOLIO_PATH) -> Portfolio:
    ticker_norm = ticker.strip().upper()
    portfolio = load_portfolio(path=path)

    holdings = [h for h in _dedupe_holdings(portfolio.holdings) if h.ticker != ticker_norm]
    new_portfolio = Portfolio(holdings=holdings)
    save_portfolio(new_portfolio, path=path)
    return new_portfolio


def _ensure_portfolio_file(path_obj: Path) -> None:
    if path_obj.exists():
        return

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    empty = Portfolio(holdings=[])
    save_portfolio(empty, path=str(path_obj))


def _dedupe_holdings(holdings: list[Holding]) -> list[Holding]:
    by_ticker: dict[str, Holding] = {}
    for holding in holdings:
        by_ticker[holding.ticker] = holding
    return sorted(by_ticker.values(), key=lambda h: h.ticker)
