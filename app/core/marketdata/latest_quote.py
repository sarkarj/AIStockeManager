from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.core.marketdata.chart_fetcher import ChartFetcher, ChartSeries
from app.core.marketdata.quotes import (
    QuoteSnapshot,
    compute_quote_display,
    get_quote_snapshot,
    infer_market_state,
)
from app.core.orchestration.time_utils import now_iso, parse_iso

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


@dataclass(frozen=True)
class LatestQuote:
    close_price: float | None
    close_ts: datetime | None
    prev_close_price: float | None
    after_hours_price: float | None
    after_hours_ts: datetime | None
    last_regular: float | None
    last_regular_ts: datetime | None
    latest_price: float | None
    latest_ts: datetime | None
    source: str  # live | cache | none
    quality_flags: set[str]
    error: str | None


def normalize_quote(raw_quote: LatestQuote | dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now_dt = now or parse_iso(now_iso())
    if isinstance(raw_quote, LatestQuote):
        payload = {
            "symbol": "",
            "currency": "USD",
            "close_price": raw_quote.close_price,
            "close_ts": raw_quote.close_ts,
            "prev_close_price": raw_quote.prev_close_price,
            "after_hours_price": raw_quote.after_hours_price,
            "after_hours_ts": raw_quote.after_hours_ts,
            "last_regular": raw_quote.last_regular,
            "last_regular_ts": raw_quote.last_regular_ts,
            "latest_price": raw_quote.latest_price,
            "latest_ts": raw_quote.latest_ts,
            "source": raw_quote.source,
            "quality_flags": set(str(flag) for flag in raw_quote.quality_flags),
            "error": raw_quote.error,
        }
    else:
        payload = dict(raw_quote or {})
        payload["quality_flags"] = set(str(flag) for flag in (payload.get("quality_flags", []) or []))

    symbol = str(payload.get("symbol") or "").strip().upper()
    currency = str(payload.get("currency") or "USD").strip().upper() or "USD"
    close_price = _to_optional_float(payload.get("close_price"))
    prev_close_price = _to_optional_float(payload.get("prev_close_price"))
    after_hours_price = _to_optional_float(payload.get("after_hours_price"))
    close_ts = _to_datetime(payload.get("close_ts"))
    after_hours_ts = _to_datetime(payload.get("after_hours_ts"))
    last_regular = _to_optional_float(payload.get("last_regular"))
    last_regular_ts = _to_datetime(payload.get("last_regular_ts"))
    snapshot = QuoteSnapshot(
        ticker=symbol,
        as_of=now_dt.astimezone(timezone.utc),
        prev_close=prev_close_price,
        close=close_price,
        close_ts=close_ts,
        after_hours=after_hours_price,
        after_hours_ts=after_hours_ts,
        last_regular=last_regular,
        last_regular_ts=last_regular_ts,
        source=str(payload.get("source") or "none"),
        quality_flags=set(payload.get("quality_flags", set())),
        error=payload.get("error"),
    )
    state = infer_market_state(
        now_et=now_dt.astimezone(_ET),
        close_ts_et=close_ts.astimezone(_ET) if isinstance(close_ts, datetime) else None,
    )
    display = compute_quote_display(snapshot, market_state=state)
    latest_price = display.latest_price
    latest_source = display.latest_label

    flags: set[str] = set(payload.get("quality_flags", set()))
    if close_price is None:
        flags.add("MISSING_CLOSE")
    if after_hours_price is None:
        flags.add("MISSING_AFTER_HOURS")
    if prev_close_price is None:
        flags.add("MISSING_PREV_CLOSE")
    if latest_price is None:
        flags.add("MISSING_LATEST")

    return {
        "symbol": symbol,
        "currency": currency,
        "close_price": close_price,
        "close_ts": close_ts.isoformat() if close_ts else None,
        "close_ts_local": _format_local_ts(close_ts),
        "after_hours_price": after_hours_price,
        "after_hours_ts": after_hours_ts.isoformat() if after_hours_ts else None,
        "after_hours_ts_local": _format_local_ts(after_hours_ts),
        "prev_close_price": prev_close_price,
        "last_regular": last_regular,
        "last_regular_ts": last_regular_ts.isoformat() if last_regular_ts else None,
        "latest_price": latest_price,
        "latest_ts": (
            (after_hours_ts if latest_source == "after_hours" else last_regular_ts or close_ts).isoformat()
            if (after_hours_ts if latest_source == "after_hours" else last_regular_ts or close_ts)
            else None
        ),
        "latest_ts_local": _format_local_ts(after_hours_ts if latest_source == "after_hours" else last_regular_ts or close_ts),
        "latest_source": latest_source,
        "today_change_abs": display.today_abs,
        "today_change_pct": (display.today_pct * 100.0) if display.today_pct is not None else None,
        "after_hours_change_abs": display.after_hours_abs,
        "after_hours_change_pct": (display.after_hours_pct * 100.0) if display.after_hours_pct is not None else None,
        "source": str(payload.get("source") or "none"),
        "quality_flags": sorted(str(flag) for flag in flags),
        "error": payload.get("error"),
    }


def get_latest_quote(
    ticker: str,
    *,
    now: datetime | None = None,
    fetcher: ChartFetcher | None = None,
    series: ChartSeries | None = None,
) -> LatestQuote:
    now_dt = now or parse_iso(now_iso())
    if isinstance(series, ChartSeries):
        return latest_quote_from_series(series=series, now=now_dt)

    snapshot = get_quote_snapshot(
        ticker=str(ticker or "").strip().upper(),
        fetcher=fetcher or ChartFetcher(cache_dir=".cache/charts"),
        now=now_dt,
    )
    state = infer_market_state(
        now_et=now_dt.astimezone(_ET),
        close_ts_et=snapshot.close_ts.astimezone(_ET) if isinstance(snapshot.close_ts, datetime) else None,
    )
    display = compute_quote_display(snapshot, market_state=state)
    latest_ts = (
        snapshot.after_hours_ts
        if display.latest_label == "after_hours"
        else snapshot.last_regular_ts or snapshot.close_ts
    )
    flags = set(str(flag) for flag in (snapshot.quality_flags or set()))
    if snapshot.close is None:
        flags.add("MISSING_CLOSE")
    if snapshot.prev_close is None:
        flags.add("MISSING_PREV_CLOSE")
    if snapshot.after_hours is None:
        flags.add("MISSING_AFTER_HOURS")
    if display.latest_price is None:
        flags.add("MISSING_LATEST")

    return LatestQuote(
        close_price=snapshot.close,
        close_ts=snapshot.close_ts,
        prev_close_price=snapshot.prev_close,
        after_hours_price=snapshot.after_hours,
        after_hours_ts=snapshot.after_hours_ts,
        last_regular=snapshot.last_regular,
        last_regular_ts=snapshot.last_regular_ts,
        latest_price=display.latest_price,
        latest_ts=latest_ts,
        source=snapshot.source,
        quality_flags=flags,
        error=snapshot.error,
    )


def latest_quote_from_series(series: ChartSeries, *, now: datetime | None = None) -> LatestQuote:
    now_dt = (now or parse_iso(now_iso())).astimezone(_ET)
    flags = {str(flag) for flag in series.quality_flags}

    if not series.bars:
        flags.add("MISSING_BARS")
        return LatestQuote(
            close_price=None,
            close_ts=None,
            prev_close_price=None,
            after_hours_price=None,
            after_hours_ts=None,
            last_regular=None,
            last_regular_ts=None,
            latest_price=None,
            latest_ts=None,
            source=str(series.source),
            quality_flags=flags,
            error=series.error,
        )

    close_price: float | None = None
    close_ts: datetime | None = None
    prev_close_price: float | None = None
    after_price: float | None = None
    after_ts: datetime | None = None

    session_closes: dict[datetime.date, tuple[datetime, float]] = {}
    candidates_after: list[tuple[datetime, float]] = []

    session_completed_dates = _completed_session_dates(series=series, now_et=now_dt)
    for bar in series.bars:
        local_ts = bar.ts.astimezone(_ET)
        if local_ts.date() in session_completed_dates and _is_rth(local_ts):
            session_closes[local_ts.date()] = (bar.ts, float(bar.close))
        if _is_after_hours(local_ts):
            candidates_after.append((bar.ts, float(bar.close)))

    close_candidates = sorted(session_closes.values(), key=lambda item: item[0])
    if close_candidates:
        close_ts, close_price = close_candidates[-1]
        if len(close_candidates) >= 2:
            _, prev_close_price = close_candidates[-2]
    else:
        flags.add("MISSING_CLOSE")

    if candidates_after:
        after_ts, after_price = sorted(candidates_after, key=lambda item: item[0])[-1]
    else:
        flags.add("MISSING_AFTER_HOURS")
    last_regular = close_price
    last_regular_ts = close_ts
    snapshot = QuoteSnapshot(
        ticker="",
        as_of=(now or parse_iso(now_iso())).astimezone(timezone.utc),
        prev_close=prev_close_price,
        close=close_price,
        close_ts=close_ts,
        after_hours=after_price,
        after_hours_ts=after_ts,
        last_regular=last_regular,
        last_regular_ts=last_regular_ts,
        source=str(series.source),
        quality_flags=flags,
        error=series.error,
    )
    state = infer_market_state(
        now_et=now_dt,
        close_ts_et=close_ts.astimezone(_ET) if isinstance(close_ts, datetime) else None,
    )
    display = compute_quote_display(snapshot, market_state=state)
    latest_ts: datetime | None = (
        after_ts if display.latest_label == "after_hours" else last_regular_ts or close_ts
    )
    if display.latest_price is None:
        flags.add("MISSING_LATEST")

    return LatestQuote(
        close_price=close_price,
        close_ts=close_ts,
        prev_close_price=prev_close_price,
        after_hours_price=after_price,
        after_hours_ts=after_ts,
        last_regular=last_regular,
        last_regular_ts=last_regular_ts,
        latest_price=display.latest_price,
        latest_ts=latest_ts,
        source=str(series.source),
        quality_flags=flags,
        error=series.error,
    )


def latest_quote_to_dict(quote: LatestQuote) -> dict[str, Any]:
    normalized = normalize_quote(quote)
    return {
        "symbol": normalized["symbol"],
        "currency": normalized["currency"],
        "close_price": quote.close_price,
        "close_ts": quote.close_ts.isoformat() if quote.close_ts else None,
        "close_ts_local": normalized["close_ts_local"],
        "prev_close_price": quote.prev_close_price,
        "last_regular": quote.last_regular,
        "last_regular_ts": quote.last_regular_ts.isoformat() if quote.last_regular_ts else None,
        "after_hours_price": quote.after_hours_price,
        "after_hours_ts": quote.after_hours_ts.isoformat() if quote.after_hours_ts else None,
        "after_hours_ts_local": normalized["after_hours_ts_local"],
        "latest_price": quote.latest_price,
        "latest_ts": quote.latest_ts.isoformat() if quote.latest_ts else None,
        "latest_ts_local": normalized["latest_ts_local"],
        "latest_source": normalized["latest_source"],
        "today_change_abs": normalized["today_change_abs"],
        "today_change_pct": normalized["today_change_pct"],
        "after_hours_change_abs": normalized["after_hours_change_abs"],
        "after_hours_change_pct": normalized["after_hours_change_pct"],
        "source": quote.source,
        "quality_flags": normalized["quality_flags"],
        "error": quote.error,
    }


def _completed_session_dates(series: ChartSeries, now_et: datetime) -> set[datetime.date]:
    dates = {bar.ts.astimezone(_ET).date() for bar in series.bars}
    if now_et.time() < _RTH_CLOSE and now_et.date() in dates:
        dates.remove(now_et.date())
    return dates


def _is_rth(ts_et: datetime) -> bool:
    t = ts_et.time()
    return _RTH_OPEN <= t <= _RTH_CLOSE


def _is_after_hours(ts_et: datetime) -> bool:
    t = ts_et.time()
    return t > _RTH_CLOSE or t < _RTH_OPEN


def _extract_official_closes(series: ChartSeries, now_et: datetime) -> tuple[float | None, datetime | None, float | None]:
    session_completed_dates = _completed_session_dates(series=series, now_et=now_et)
    session_closes: dict[datetime.date, tuple[datetime, float]] = {}
    for bar in series.bars:
        local_ts = bar.ts.astimezone(_ET)
        if local_ts.date() not in session_completed_dates:
            continue
        if not _is_rth(local_ts):
            continue
        session_closes[local_ts.date()] = (bar.ts, float(bar.close))

    sorted_closes = [session_closes[key] for key in sorted(session_closes.keys())]
    if not sorted_closes:
        return None, None, None
    close_ts, close_price = sorted_closes[-1]
    prev_close = sorted_closes[-2][1] if len(sorted_closes) >= 2 else None
    return close_price, close_ts, prev_close


def _combine_source(primary: str, secondary: str) -> str:
    states = {str(primary or "").lower(), str(secondary or "").lower()}
    if "live" in states:
        return "live"
    if "cache" in states:
        return "cache"
    return "none"


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _format_local_ts(value: datetime | None) -> str | None:
    if not isinstance(value, datetime):
        return None
    local = value.astimezone(_ET)
    return local.strftime("%Y-%m-%d %H:%M ET")
