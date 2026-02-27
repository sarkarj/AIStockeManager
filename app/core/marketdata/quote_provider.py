from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.time_utils import minutes_between, now_iso, parse_iso

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Quote:
    close_price: float | None
    close_ts: datetime | None
    prev_close_price: float | None
    after_hours_price: float | None
    after_hours_ts: datetime | None
    currency: str
    source: str  # live | cache | none
    quality_flags: set[str]
    error: str | None


@dataclass
class _QuoteComponent:
    bars: list[dict[str, Any]]
    source: str  # live | cache | none
    stale_cache: bool
    cache_hit: bool
    cache_age_minutes: float | None
    error: str | None
    attempts: int


class QuoteProvider:
    """Canonical quote source for UI display pricing (close + after-hours)."""

    def __init__(self, cache_dir: str = ".cache/quotes") -> None:
        self.cache = DiskTTLCache(base_dir=cache_dir)
        self.cache_dir = cache_dir

    def get_quote(self, ticker: str, *, now: datetime | None = None) -> Quote:
        ticker_norm = str(ticker or "").strip().upper()
        if not ticker_norm:
            return Quote(
                close_price=None,
                close_ts=None,
                prev_close_price=None,
                after_hours_price=None,
                after_hours_ts=None,
                currency="USD",
                source="none",
                quality_flags={"MISSING_CLOSE", "MISSING_AFTER_HOURS"},
                error="empty_ticker",
            )

        now_dt = now or parse_iso(now_iso())
        now_iso_value = now_dt.isoformat()

        close_component = self._fetch_component(
            ticker=ticker_norm,
            component="close",
            period="5d",
            interval="1d",
            prepost=False,
            ttl_seconds=3600,
            now_iso_value=now_iso_value,
        )
        after_component = self._fetch_component(
            ticker=ticker_norm,
            component="after_hours",
            period="1d",
            interval="5m",
            prepost=True,
            ttl_seconds=300,
            now_iso_value=now_iso_value,
        )

        close_price, close_ts, prev_close = _extract_close_fields(close_component.bars)
        after_price, after_ts = _extract_after_hours_fields(after_component.bars)

        flags: set[str] = set()
        if close_price is None:
            flags.add("MISSING_CLOSE")
        if after_price is None:
            flags.add("MISSING_AFTER_HOURS")
        elif _is_regular_session(after_ts):
            flags.add("MAY_BE_RTH_ONLY")

        if close_component.stale_cache or after_component.stale_cache:
            flags.add("STALE")

        if close_component.source == "live" or after_component.source == "live":
            source = "live"
        elif close_component.source == "cache" or after_component.source == "cache":
            source = "cache"
        else:
            source = "none"

        error = _first_non_empty([close_component.error, after_component.error])

        return Quote(
            close_price=close_price,
            close_ts=close_ts,
            prev_close_price=prev_close,
            after_hours_price=after_price,
            after_hours_ts=after_ts,
            currency="USD",
            source=source,
            quality_flags=flags,
            error=error,
        )

    def _fetch_component(
        self,
        *,
        ticker: str,
        component: str,
        period: str,
        interval: str,
        prepost: bool,
        ttl_seconds: int,
        now_iso_value: str,
    ) -> _QuoteComponent:
        cache_key = f"quote:{component}:{ticker}:{period}:{interval}:{int(prepost)}"
        cache_path = self.cache.path_for_key(cache_key)

        cached_fresh = self.cache.get(cache_key, now_iso=now_iso_value)
        if isinstance(cached_fresh, dict):
            bars = _bars_from_payload(cached_fresh)
            if bars:
                return _QuoteComponent(
                    bars=bars,
                    source="cache",
                    stale_cache=False,
                    cache_hit=True,
                    cache_age_minutes=_cache_age_minutes(cache_path, now_iso_value),
                    error=None,
                    attempts=0,
                )
            _delete_cache_file(cache_path)

        last_error: str | None = None
        attempts = 0
        for delay in [0.0, 0.5, 1.0, 2.0]:
            if delay > 0:
                time.sleep(delay)
            attempts += 1
            try:
                frame = self._history_with_yfinance(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    prepost=prepost,
                )
                bars_live = _bars_from_frame(frame)
                if bars_live:
                    payload = {
                        "as_of": bars_live[-1]["ts"],
                        "bars": bars_live,
                    }
                    self.cache.set(cache_key, payload, ttl_seconds=ttl_seconds)
                    return _QuoteComponent(
                        bars=bars_live,
                        source="live",
                        stale_cache=False,
                        cache_hit=False,
                        cache_age_minutes=0.0,
                        error=None,
                        attempts=attempts,
                    )
                last_error = "empty_live"
            except Exception as exc:  # noqa: PERF203
                last_error = _one_line_error(str(exc))

        stale_payload, stale_cache = self._read_any_cache_payload(
            cache_key=cache_key,
            now_iso_value=now_iso_value,
            ttl_seconds=ttl_seconds,
        )
        if isinstance(stale_payload, dict):
            bars = _bars_from_payload(stale_payload)
            if bars:
                return _QuoteComponent(
                    bars=bars,
                    source="cache",
                    stale_cache=bool(stale_cache),
                    cache_hit=True,
                    cache_age_minutes=_cache_age_minutes(cache_path, now_iso_value),
                    error=last_error,
                    attempts=attempts,
                )

        return _QuoteComponent(
            bars=[],
            source="none",
            stale_cache=False,
            cache_hit=False,
            cache_age_minutes=None,
            error=last_error or "empty_live",
            attempts=attempts,
        )

    def _read_any_cache_payload(
        self,
        *,
        cache_key: str,
        now_iso_value: str,
        ttl_seconds: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        path = Path(self.cache.path_for_key(cache_key))
        if not path.exists():
            return None, False
        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, False

        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None, False

        cached_at = record.get("cached_at")
        record_ttl = record.get("ttl_seconds")
        stale = False
        if isinstance(cached_at, str):
            try:
                age_seconds = (parse_iso(now_iso_value) - parse_iso(cached_at)).total_seconds()
                effective_ttl = int(ttl_seconds)
                if isinstance(record_ttl, int):
                    effective_ttl = int(record_ttl)
                stale = effective_ttl <= 0 or age_seconds > effective_ttl
            except Exception:
                stale = False
        return payload, stale

    def _history_with_yfinance(self, ticker: str, period: str, interval: str, prepost: bool) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("yfinance_unavailable") from exc

        tk = yf.Ticker(ticker)
        frame = tk.history(period=period, interval=interval, prepost=prepost)
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError("invalid_yfinance_response")
        return frame


def quote_to_dict(quote: Quote) -> dict[str, Any]:
    return {
        "close_price": quote.close_price,
        "close_ts": quote.close_ts.isoformat() if isinstance(quote.close_ts, datetime) else None,
        "prev_close_price": quote.prev_close_price,
        "after_hours_price": quote.after_hours_price,
        "after_hours_ts": quote.after_hours_ts.isoformat() if isinstance(quote.after_hours_ts, datetime) else None,
        "currency": quote.currency,
        "source": quote.source,
        "quality_flags": sorted(str(flag) for flag in quote.quality_flags),
        "error": quote.error,
    }


def quote_latest_price(quote: Quote | dict[str, Any] | None) -> float | None:
    if quote is None:
        return None
    if isinstance(quote, Quote):
        return quote.after_hours_price if quote.after_hours_price is not None else quote.close_price
    after_price = _to_optional_float(quote.get("after_hours_price"))
    close_price = _to_optional_float(quote.get("close_price"))
    return after_price if after_price is not None else close_price


def _extract_close_fields(bars: list[dict[str, Any]]) -> tuple[float | None, datetime | None, float | None]:
    if not bars:
        return None, None, None
    last = bars[-1]
    close_price = _to_optional_float(last.get("close"))
    close_ts = _to_datetime(last.get("ts"))
    prev_close: float | None = None
    if len(bars) >= 2:
        prev_close = _to_optional_float(bars[-2].get("close"))
    return close_price, close_ts, prev_close


def _extract_after_hours_fields(bars: list[dict[str, Any]]) -> tuple[float | None, datetime | None]:
    if not bars:
        return None, None
    last = bars[-1]
    return _to_optional_float(last.get("close")), _to_datetime(last.get("ts"))


def _is_regular_session(ts: datetime | None) -> bool:
    if not isinstance(ts, datetime):
        return False
    local = ts.astimezone(_ET)
    minutes = local.hour * 60 + local.minute
    start = 9 * 60 + 30
    end = 16 * 60
    return start <= minutes <= end


def _bars_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
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

    bars: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        bars.append(
            {
                "ts": _to_datetime(row.ts).isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": None if pd.isna(row.volume) else float(row.volume),
            }
        )
    return bars


def _bars_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bars_raw = payload.get("bars", [])
    if not isinstance(bars_raw, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in bars_raw:
        if not isinstance(item, dict):
            continue
        ts = _to_datetime(item.get("ts"))
        close = _to_optional_float(item.get("close"))
        open_px = _to_optional_float(item.get("open"))
        high = _to_optional_float(item.get("high"))
        low = _to_optional_float(item.get("low"))
        if ts is None or close is None or open_px is None or high is None or low is None:
            continue
        parsed.append(
            {
                "ts": ts.isoformat(),
                "open": float(open_px),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": _to_optional_float(item.get("volume")),
            }
        )
    parsed.sort(key=lambda bar: bar["ts"])
    deduped: dict[str, dict[str, Any]] = {}
    for bar in parsed:
        deduped[str(bar["ts"])] = bar
    return [deduped[key] for key in sorted(deduped.keys())]


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


def _delete_cache_file(cache_path: str) -> None:
    path = Path(cache_path)
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        return


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
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


def _one_line_error(text: str) -> str:
    compact = " ".join(str(text).split())
    return compact[:180]


def _first_non_empty(values: list[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None
