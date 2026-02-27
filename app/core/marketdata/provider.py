from __future__ import annotations

from typing import Protocol


class MarketDataProvider(Protocol):
    def get_ohlcv(self, ticker: str, interval: str, lookback_days: int) -> dict:
        """Return {'as_of': ISO, 'bars': [{ts, open, high, low, close, volume}]}"""
