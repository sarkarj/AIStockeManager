# DRL Indicator Feature Contract (Phase 2)

- `price_last`: latest close price (currency units).
- `ema_50`: 50-period exponential moving average of close (currency units).
- `sma_200`: 200-period simple moving average of close (currency units).
- `rsi_14`: 14-period RSI in range `[0, 100]`.
- `macd`: MACD line (`EMA12 - EMA26`, currency units).
- `macd_signal`: 9-period EMA of MACD line (currency units).
- `stoch_k`: stochastic `%K` over 14 periods in range `[0, 100]`.
- `adx_14`: 14-period ADX trend strength in range `[0, 100]`.
- `vroc_14`: 14-period volume rate-of-change in percent.
- `atr_pct`: ATR as percentage of close (`ATR14 / close * 100`).

Phase 2 temporary proxy flags:
- `supertrend_dir_1D = "BULL" if price_last > ema_50 else "BEAR"`
- `supertrend_dir_1W = "BULL" if price_last > sma_200 else "BEAR"`

These two flags are placeholders until a true Supertrend implementation is added in a later phase.
