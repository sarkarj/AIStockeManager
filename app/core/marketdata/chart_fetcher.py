from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import minutes_between, now_iso, parse_iso

RANGE_MAPPING: dict[str, dict[str, Any]] = {
    "1D": {"period": "1d", "interval": "5m", "prepost": True, "ttl_seconds": 300, "expected_tz": "America/New_York"},
    "1W": {"period": "5d", "interval": "30m", "prepost": True, "ttl_seconds": 3600, "expected_tz": "America/New_York"},
    "1M": {"period": "1mo", "interval": "1h", "prepost": True, "ttl_seconds": 3600, "expected_tz": "America/New_York"},
    "3M": {"period": "3mo", "interval": "1d", "prepost": False, "ttl_seconds": 3600, "expected_tz": "America/New_York"},
    "YTD": {"period": "ytd", "interval": "1d", "prepost": False, "ttl_seconds": 3600, "expected_tz": "America/New_York"},
    "1Y": {"period": "1y", "interval": "1d", "prepost": False, "ttl_seconds": 3600, "expected_tz": "America/New_York"},
}

_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass
class ChartSeries:
    bars: list[Bar]
    as_of: datetime
    source: str  # live | cache | none
    error: str | None
    quality_flags: set[str]
    cache_path: str
    cache_age_minutes: float | None
    cache_hit: bool
    stale_cache: bool
    attempts: int


class ChartFetcher:
    _FAILURE_UNTIL: dict[str, float] = {}

    def __init__(
        self,
        cache_dir: str = ".cache/charts",
        *,
        stale_first: bool | None = None,
        cache_only: bool | None = None,
        failure_cooldown_seconds: int = 60,
    ):
        self.cache = DiskTTLCache(base_dir=cache_dir)
        self.cache_dir = cache_dir
        if stale_first is None:
            stale_first = str(os.getenv("MARKETDATA_STALE_FIRST", "0")).strip().lower() not in {"0", "false", "no"}
        if cache_only is None:
            cache_only = str(os.getenv("MARKETDATA_CACHE_ONLY", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self.stale_first = bool(stale_first)
        self.cache_only = bool(cache_only)
        self.failure_cooldown_seconds = max(1, int(failure_cooldown_seconds))

    def fetch_chart_series(self, ticker: str, range_key: str, *, force_revalidate: bool = False) -> ChartSeries:
        mapping = range_mapping(range_key)
        cache_key = self._cache_key(ticker=ticker, range_key=range_key, mapping=mapping)
        cache_path = self.cache.path_for_key(cache_key)
        cooldown_key = self._cooldown_key(ticker=ticker, range_key=range_key, mapping=mapping)
        request_now = now_iso()
        empty_cache_seen = False

        if not force_revalidate:
            cached_fresh = self.cache.get(cache_key, now_iso=request_now)
            if isinstance(cached_fresh, dict):
                cached_bars = _bars_from_payload(cached_fresh)
                if cached_bars:
                    return ChartSeries(
                        bars=cached_bars,
                        as_of=cached_bars[-1].ts,
                        source="cache",
                        error=None,
                        quality_flags=set(),
                        cache_path=cache_path,
                        cache_age_minutes=_cache_age_minutes(cache_path=cache_path, now_iso_value=request_now),
                        cache_hit=True,
                        stale_cache=False,
                        attempts=0,
                    )
                empty_cache_seen = True
                _delete_cache_file(cache_path)

        cached_any, stale_cache = self._read_any_cache_payload(cache_key=cache_key, now_iso_value=request_now)
        cache_bars_any = _bars_from_payload(cached_any) if isinstance(cached_any, dict) else []
        cache_age_minutes_any = _cache_age_minutes(cache_path=cache_path, now_iso_value=request_now) if cache_bars_any else None
        ttl_minutes = float(mapping["ttl_seconds"]) / 60.0
        stale_cache_by_age = (
            isinstance(cache_age_minutes_any, (int, float))
            and cache_age_minutes_any > ttl_minutes
        )
        stale_cache_effective_any = bool(stale_cache or stale_cache_by_age)

        cooldown_active = self._cooldown_active(cooldown_key)
        if cache_bars_any and not force_revalidate and (self.stale_first or cooldown_active):
            flags = {"STALE_CACHE"} if stale_cache_effective_any else set()
            if cooldown_active:
                flags.add("LIVE_COOLDOWN")
            if self.cache_only:
                flags.add("CACHE_ONLY")
            return ChartSeries(
                bars=cache_bars_any,
                as_of=cache_bars_any[-1].ts,
                source="cache",
                error="live_cooldown" if cooldown_active else None,
                quality_flags=flags,
                cache_path=cache_path,
                cache_age_minutes=cache_age_minutes_any,
                cache_hit=True,
                stale_cache=stale_cache_effective_any,
                attempts=0,
            )

        if self.cache_only and not force_revalidate:
            if cache_bars_any:
                flags = {"CACHE_ONLY"}
                if stale_cache_effective_any:
                    flags.add("STALE_CACHE")
                return ChartSeries(
                    bars=cache_bars_any,
                    as_of=cache_bars_any[-1].ts,
                    source="cache",
                    error=None,
                    quality_flags=flags,
                    cache_path=cache_path,
                    cache_age_minutes=cache_age_minutes_any,
                    cache_hit=True,
                    stale_cache=stale_cache_effective_any,
                    attempts=0,
                )
            return ChartSeries(
                bars=[],
                as_of=parse_iso(request_now),
                source="none",
                error="cache_only_miss",
                quality_flags={"MISSING", "CACHE_ONLY"},
                cache_path=cache_path,
                cache_age_minutes=None,
                cache_hit=False,
                stale_cache=False,
                attempts=0,
            )

        last_error: str | None = None
        attempts = 0
        for delay in [0.0, 0.5, 1.0, 2.0]:
            if delay > 0:
                time.sleep(delay)
            attempts += 1
            try:
                frame = self._history_with_yfinance(
                    ticker=str(ticker).strip().upper(),
                    period=str(mapping["period"]),
                    interval=str(mapping["interval"]),
                    prepost=bool(mapping["prepost"]),
                )
                bars_live = _bars_from_frame(frame)
                if bars_live:
                    payload = {
                        "as_of": bars_live[-1].ts.isoformat(),
                        "bars": [bar_to_dict(bar) for bar in bars_live],
                    }
                    self.cache.set(cache_key, payload, ttl_seconds=int(mapping["ttl_seconds"]))
                    self._clear_cooldown(cooldown_key)
                    return ChartSeries(
                        bars=bars_live,
                        as_of=bars_live[-1].ts,
                        source="live",
                        error=None,
                        quality_flags=set(),
                        cache_path=cache_path,
                        cache_age_minutes=0.0,
                        cache_hit=False,
                        stale_cache=False,
                        attempts=attempts,
                    )
                last_error = "empty_live"
            except Exception as exc:  # noqa: PERF203
                last_error = _one_line_error(str(exc))
                if _is_non_retryable_error(last_error):
                    break

        self._mark_failure(cooldown_key)

        if isinstance(cached_any, dict):
            cache_bars = cache_bars_any
            cache_age_minutes = cache_age_minutes_any
            stale_cache_effective = stale_cache_effective_any
            flags = {"EMPTY_LIVE"}
            if stale_cache_effective:
                flags.add("STALE_CACHE")
            if not cache_bars:
                flags.add("EMPTY_CACHE")
                empty_cache_seen = True
                _delete_cache_file(cache_path)
            if cache_bars:
                return ChartSeries(
                    bars=cache_bars,
                    as_of=cache_bars[-1].ts,
                    source="cache",
                    error=last_error,
                    quality_flags=flags,
                    cache_path=cache_path,
                    cache_age_minutes=cache_age_minutes,
                    cache_hit=True,
                    stale_cache=stale_cache_effective,
                    attempts=attempts,
                )

        return ChartSeries(
            bars=[],
            as_of=parse_iso(request_now),
            source="none",
            error=last_error or "empty_live",
            quality_flags={"EMPTY_LIVE", "MISSING"} | ({"EMPTY_CACHE"} if empty_cache_seen else set()),
            cache_path=cache_path,
            cache_age_minutes=None,
            cache_hit=False,
            stale_cache=False,
            attempts=attempts,
        )

    def _cooldown_key(self, ticker: str, range_key: str, mapping: dict[str, Any]) -> str:
        ticker_norm = str(ticker).strip().upper()
        return f"{ticker_norm}:{range_key}:{mapping['period']}:{mapping['interval']}:{int(bool(mapping['prepost']))}"

    def _cooldown_active(self, key: str) -> bool:
        until = float(self._FAILURE_UNTIL.get(key, 0.0) or 0.0)
        now_ts = time.time()
        if until <= now_ts:
            self._FAILURE_UNTIL.pop(key, None)
            return False
        return True

    def _mark_failure(self, key: str) -> None:
        self._FAILURE_UNTIL[key] = time.time() + float(self.failure_cooldown_seconds)

    def _clear_cooldown(self, key: str) -> None:
        self._FAILURE_UNTIL.pop(key, None)

    def _cache_key(self, ticker: str, range_key: str, mapping: dict[str, Any]) -> str:
        ticker_norm = str(ticker).strip().upper()
        return (
            f"brain-chart:{ticker_norm}:{range_key}:{mapping['period']}:"
            f"{mapping['interval']}:{int(bool(mapping['prepost']))}"
        )

    def _history_with_yfinance(self, ticker: str, period: str, interval: str, prepost: bool) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception:
            return self._history_with_yahoo_chart_api(
                ticker=ticker,
                period=period,
                interval=interval,
                prepost=prepost,
            )

        try:
            tk = yf.Ticker(ticker)
            frame = tk.history(period=period, interval=interval, prepost=prepost)
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError("invalid_yfinance_response")
            return frame
        except Exception:
            return self._history_with_yahoo_chart_api(
                ticker=ticker,
                period=period,
                interval=interval,
                prepost=prepost,
            )

    def _history_with_yahoo_chart_api(self, ticker: str, period: str, interval: str, prepost: bool) -> pd.DataFrame:
        end_dt = datetime.now(timezone.utc)
        lookback_days = _period_to_lookback_days(period=period, end_dt=end_dt)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        params = {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": interval,
            "includePrePost": "true" if prepost else "false",
            "events": "div,splits",
        }
        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for base_url in (
            "https://query1.finance.yahoo.com/v8/finance/chart",
            "https://query2.finance.yahoo.com/v8/finance/chart",
        ):
            url = f"{base_url}/{str(ticker).strip().upper()}"
            try:
                response = requests.get(url, params=params, timeout=10, headers=_YAHOO_HEADERS)
                response.raise_for_status()
                payload = response.json() or {}
                break
            except Exception as exc:
                last_error = exc
                continue
        if payload is None:
            raise RuntimeError(f"provider_unavailable:{_one_line_error(str(last_error))}") from last_error

        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RuntimeError("provider_unavailable:chart_error")
        results = chart.get("result", [])
        if not results:
            raise RuntimeError("no_yahoo_results")

        result = results[0]
        ts_values = result.get("timestamp", []) or []
        quote_items = ((result.get("indicators") or {}).get("quote") or [])
        if not ts_values or not quote_items:
            raise RuntimeError("invalid_yahoo_response")

        quote = quote_items[0]
        opens = quote.get("open", []) or []
        highs = quote.get("high", []) or []
        lows = quote.get("low", []) or []
        closes = quote.get("close", []) or []
        volumes = quote.get("volume", []) or []

        rows: list[dict[str, Any]] = []
        for idx, ts_val in enumerate(ts_values):
            if idx >= len(opens) or idx >= len(highs) or idx >= len(lows) or idx >= len(closes):
                continue
            o = opens[idx]
            h = highs[idx]
            l = lows[idx]
            c = closes[idx]
            if o is None or h is None or l is None or c is None:
                continue
            volume = volumes[idx] if idx < len(volumes) else None
            rows.append(
                {
                    "ts": datetime.fromtimestamp(int(ts_val), tz=timezone.utc),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": None if volume is None else float(volume),
                }
            )

        if not rows:
            raise RuntimeError("empty_live")
        return pd.DataFrame(rows)

    def _read_any_cache_payload(self, cache_key: str, now_iso_value: str) -> tuple[dict[str, Any] | None, bool]:
        path = Path(self.cache.path_for_key(cache_key))
        if not path.exists():
            return None, False
        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, False

        payload = record.get("payload")
        cached_at = record.get("cached_at")
        ttl_seconds = record.get("ttl_seconds")
        if not isinstance(payload, dict):
            return None, False

        stale_cache = False
        if isinstance(cached_at, str) and isinstance(ttl_seconds, int):
            try:
                age_seconds = (parse_iso(now_iso_value) - parse_iso(cached_at)).total_seconds()
                stale_cache = age_seconds > int(ttl_seconds)
            except Exception:
                stale_cache = False
        return payload, stale_cache


def range_mapping(range_key: str, advanced_source_range: str | None = None) -> dict[str, Any]:
    key = str(range_key).upper()
    if key == "ADVANCED":
        adv = str(advanced_source_range or "3M").upper()
        key = adv if adv in RANGE_MAPPING else "3M"
    mapping = RANGE_MAPPING.get(key, RANGE_MAPPING["3M"])
    return dict(mapping)


def bar_to_dict(bar: Bar) -> dict[str, Any]:
    return {
        "ts": bar.ts.isoformat(),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": None if bar.volume is None else float(bar.volume),
    }


def _bars_from_payload(payload: dict[str, Any]) -> list[Bar]:
    bars = payload.get("bars", [])
    if not isinstance(bars, list):
        return []
    parsed: list[Bar] = []
    for item in bars:
        if not isinstance(item, dict):
            continue
        ts_raw = item.get("ts")
        try:
            ts = _to_datetime(ts_raw)
            parsed.append(
                Bar(
                    ts=ts,
                    open=float(item.get("open")),
                    high=float(item.get("high")),
                    low=float(item.get("low")),
                    close=float(item.get("close")),
                    volume=_to_optional_float(item.get("volume")),
                )
            )
        except Exception:
            continue
    parsed.sort(key=lambda bar: bar.ts)
    deduped: dict[datetime, Bar] = {}
    for bar in parsed:
        deduped[bar.ts] = bar
    return [deduped[key] for key in sorted(deduped.keys())]


def _bars_from_frame(frame: pd.DataFrame) -> list[Bar]:
    if frame.empty:
        return []

    work = frame.copy()
    work = work.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "Datetime" in work.columns:
        work = work.rename(columns={"Datetime": "ts"})
    if "Date" in work.columns:
        work = work.rename(columns={"Date": "ts"})
    if "ts" not in work.columns:
        work = work.reset_index().rename(columns={work.index.name or "index": "ts"})
        if "ts" not in work.columns:
            work = work.rename(columns={work.columns[0]: "ts"})

    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = None
    else:
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce")

    work = work.dropna(subset=["ts", "open", "high", "low", "close"])
    if work.empty:
        return []
    work = work.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    bars: list[Bar] = []
    for row in work.itertuples(index=False):
        bars.append(
            Bar(
                ts=_to_datetime(row.ts),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=None if pd.isna(row.volume) else float(row.volume),
            )
        )
    return bars


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_age_minutes(cache_path: str, now_iso_value: str) -> float | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    cached_at = record.get("cached_at")
    if not isinstance(cached_at, str):
        return None
    try:
        return max(0.0, float(minutes_between(cached_at, now_iso_value)))
    except Exception:
        return None


def _one_line_error(text: str) -> str:
    compact = " ".join(str(text).split())
    return compact[:180]


def _delete_cache_file(cache_path: str) -> None:
    path = Path(cache_path)
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        return


def _period_to_lookback_days(period: str, end_dt: datetime) -> int:
    value = str(period or "").strip().lower()
    if value.endswith("d"):
        try:
            return max(2, int(value[:-1]) + 2)
        except ValueError:
            return 7
    if value.endswith("mo"):
        try:
            return max(7, int(value[:-2]) * 31 + 3)
        except ValueError:
            return 32
    if value.endswith("y"):
        try:
            return max(60, int(value[:-1]) * 366 + 3)
        except ValueError:
            return 370
    if value == "ytd":
        jan1 = datetime(end_dt.year, 1, 1, tzinfo=timezone.utc)
        days = (end_dt - jan1).days + 3
        return max(7, int(days))
    return 7


def _is_non_retryable_error(error: str | None) -> bool:
    value = str(error or "").lower()
    if not value:
        return False
    prefixes = (
        "provider_unavailable",
        "invalid_yahoo_response",
        "no_yahoo_results",
        "yfinance_unavailable",
    )
    return any(value.startswith(prefix) for prefix in prefixes)
