from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from collections.abc import Callable
from typing import Literal
from zoneinfo import ZoneInfo

from app.core.marketdata.chart_fetcher import ChartFetcher, ChartSeries
from app.core.orchestration.time_utils import now_iso, parse_iso

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)

MarketState = Literal["open", "closed", "unknown"]


@dataclass(frozen=True)
class QuoteSnapshot:
    ticker: str
    as_of: datetime
    prev_close: float | None = None
    close: float | None = None
    close_ts: datetime | None = None
    after_hours: float | None = None
    after_hours_ts: datetime | None = None
    last_regular: float | None = None
    last_regular_ts: datetime | None = None
    session_state: str = "unknown"
    source: str = "none"
    quality_flags: set[str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class QuoteSnapshotDisplay:
    latest_price: float | None
    latest_label: Literal["after_hours", "regular"]
    today_abs: float | None
    today_pct: float | None
    after_hours_abs: float | None
    after_hours_pct: float | None


def quote_snapshot_to_dict(q: QuoteSnapshot, *, market_state: MarketState | None = None) -> dict[str, object]:
    state = market_state or infer_market_state(
        now_et=q.as_of.astimezone(_ET),
        close_ts_et=q.close_ts.astimezone(_ET) if isinstance(q.close_ts, datetime) else None,
    )
    display = compute_quote_display(q, market_state=state)
    latest_ts = (
        q.after_hours_ts
        if display.latest_label == "after_hours"
        else q.last_regular_ts or q.close_ts
    )
    flags = sorted(str(flag) for flag in (q.quality_flags or set()))
    session_state = _session_state_label(q.as_of.astimezone(_ET))
    extended_label = _extended_label_for_session(session_state=session_state)
    show_extended_session = bool(extended_label and q.after_hours is not None)
    return {
        "symbol": q.ticker,
        "currency": "USD",
        "close_price": q.close,
        "close_ts": q.close_ts.isoformat() if q.close_ts else None,
        "close_ts_local": _format_local_ts(q.close_ts),
        "prev_close_price": q.prev_close,
        "last_regular": q.last_regular,
        "last_regular_ts": q.last_regular_ts.isoformat() if q.last_regular_ts else None,
        "after_hours_price": q.after_hours,
        "after_hours_ts": q.after_hours_ts.isoformat() if q.after_hours_ts else None,
        "after_hours_ts_local": _format_local_ts(q.after_hours_ts),
        "latest_price": display.latest_price,
        "latest_ts": latest_ts.isoformat() if latest_ts else None,
        "latest_ts_local": _format_local_ts(latest_ts),
        "latest_source": display.latest_label,
        "today_change_abs": display.today_abs,
        "today_change_pct": (display.today_pct * 100.0) if display.today_pct is not None else None,
        "after_hours_change_abs": display.after_hours_abs,
        "after_hours_change_pct": (display.after_hours_pct * 100.0) if display.after_hours_pct is not None else None,
        "session_state": session_state,
        "show_extended_session": show_extended_session,
        "extended_label": extended_label,
        "display_price_source": "latest" if display.latest_price is not None else "close",
        "source": q.source,
        "as_of_ts": q.as_of.isoformat(),
        "quality_flags": flags,
        "error": q.error,
    }


def compute_quote_display(
    q: QuoteSnapshot,
    *,
    market_state: MarketState,
) -> QuoteSnapshotDisplay:
    latest_price: float | None = None
    latest_label: Literal["after_hours", "regular"] = "regular"

    if market_state == "closed" and q.after_hours is not None and q.close is not None:
        latest_price = q.after_hours
        latest_label = "after_hours"
    elif q.last_regular is not None:
        latest_price = q.last_regular
        latest_label = "regular"
    elif q.close is not None:
        latest_price = q.close
        latest_label = "regular"
    elif q.after_hours is not None and market_state == "unknown":
        latest_price = q.after_hours
        latest_label = "after_hours"

    today_abs: float | None = None
    today_pct: float | None = None
    if q.prev_close is not None and q.close is not None and q.prev_close != 0:
        today_abs = q.close - q.prev_close
        today_pct = today_abs / q.prev_close

    after_hours_abs: float | None = None
    after_hours_pct: float | None = None
    if q.close is not None and q.after_hours is not None and q.close != 0:
        after_hours_abs = q.after_hours - q.close
        after_hours_pct = after_hours_abs / q.close

    return QuoteSnapshotDisplay(
        latest_price=latest_price,
        latest_label=latest_label,
        today_abs=today_abs,
        today_pct=today_pct,
        after_hours_abs=after_hours_abs,
        after_hours_pct=after_hours_pct,
    )


def infer_market_state(now_et: datetime, close_ts_et: datetime | None) -> MarketState:
    if close_ts_et is None:
        return "unknown"
    now_local = now_et.astimezone(_ET)
    if now_local.weekday() >= 5:
        return "closed"
    t = now_local.time()
    if _RTH_OPEN <= t < _RTH_CLOSE:
        return "open"
    return "closed"


def get_quote_snapshot(
    ticker: str,
    *,
    fetcher: ChartFetcher | None = None,
    series_resolver: Callable[[str], ChartSeries | None] | None = None,
    now: datetime | None = None,
) -> QuoteSnapshot:
    ticker_norm = str(ticker or "").strip().upper()
    now_dt = (now or parse_iso(now_iso())).astimezone(timezone.utc)
    fetcher_obj = fetcher or ChartFetcher(cache_dir=".cache/charts")
    flags: set[str] = set()
    errors: list[str] = []

    series_1d = _safe_fetch(
        fetcher=fetcher_obj,
        ticker=ticker_norm,
        range_key="1D",
        errors=errors,
        series_resolver=series_resolver,
    )
    series_1w = _safe_fetch(
        fetcher=fetcher_obj,
        ticker=ticker_norm,
        range_key="1W",
        errors=errors,
        series_resolver=series_resolver,
    )

    close, close_ts, prev_close = _regular_close_from_series(series_1w, now_et=now_dt.astimezone(_ET))
    last_regular, last_regular_ts = _latest_regular_from_intraday(series_1d)
    after_hours, after_hours_ts = _latest_after_hours_from_intraday(series_1d)

    if close is None:
        close = last_regular
        close_ts = last_regular_ts

    source = _combine_source(
        primary=series_1d.source if isinstance(series_1d, ChartSeries) else "none",
        secondary=series_1w.source if isinstance(series_1w, ChartSeries) else "none",
    )

    if isinstance(series_1d, ChartSeries):
        flags.update(str(flag) for flag in series_1d.quality_flags)
    if isinstance(series_1w, ChartSeries):
        flags.update(str(flag) for flag in series_1w.quality_flags)
    if close is None:
        flags.add("MISSING_CLOSE")
    if prev_close is None:
        flags.add("MISSING_PREV_CLOSE")
    if after_hours is None:
        flags.add("MISSING_AFTER_HOURS")
    if last_regular is None and close is None:
        flags.add("MISSING_REGULAR")

    error = next((e for e in errors if e), None)
    session_state = _session_state_label(now_dt.astimezone(_ET))
    return QuoteSnapshot(
        ticker=ticker_norm,
        as_of=now_dt,
        prev_close=prev_close,
        close=close,
        close_ts=close_ts,
        after_hours=after_hours,
        after_hours_ts=after_hours_ts,
        last_regular=last_regular,
        last_regular_ts=last_regular_ts,
        session_state=session_state,
        source=source,
        quality_flags=flags,
        error=error,
    )


def _safe_fetch(
    fetcher: ChartFetcher,
    ticker: str,
    range_key: str,
    *,
    errors: list[str],
    series_resolver: Callable[[str], ChartSeries | None] | None = None,
) -> ChartSeries | None:
    try:
        if callable(series_resolver):
            resolved = series_resolver(str(range_key).upper())
            if isinstance(resolved, ChartSeries):
                return resolved
        return fetcher.fetch_chart_series(ticker=ticker, range_key=range_key)
    except Exception as exc:  # noqa: PERF203
        errors.append(str(exc))
        return None


def _latest_regular_from_intraday(series: ChartSeries | None) -> tuple[float | None, datetime | None]:
    if not isinstance(series, ChartSeries):
        return None, None
    candidates: list[tuple[datetime, float]] = []
    for bar in series.bars:
        local = bar.ts.astimezone(_ET)
        if _RTH_OPEN <= local.time() <= _RTH_CLOSE:
            candidates.append((bar.ts, float(bar.close)))
    if not candidates:
        return None, None
    ts, value = sorted(candidates, key=lambda item: item[0])[-1]
    return value, ts


def _latest_after_hours_from_intraday(series: ChartSeries | None) -> tuple[float | None, datetime | None]:
    if not isinstance(series, ChartSeries):
        return None, None
    candidates: list[tuple[datetime, float]] = []
    for bar in series.bars:
        local = bar.ts.astimezone(_ET)
        if local.time() > _RTH_CLOSE or local.time() < _RTH_OPEN:
            candidates.append((bar.ts, float(bar.close)))
    if not candidates:
        return None, None
    ts, value = sorted(candidates, key=lambda item: item[0])[-1]
    return value, ts


def _regular_close_from_series(
    series: ChartSeries | None,
    *,
    now_et: datetime,
) -> tuple[float | None, datetime | None, float | None]:
    if not isinstance(series, ChartSeries):
        return None, None, None

    session_closes: dict[datetime.date, tuple[datetime, float]] = {}
    for bar in series.bars:
        local = bar.ts.astimezone(_ET)
        if not (_RTH_OPEN <= local.time() <= _RTH_CLOSE):
            continue
        if now_et.time() < _RTH_CLOSE and local.date() == now_et.date():
            continue
        session_closes[local.date()] = (bar.ts, float(bar.close))

    sorted_closes = [session_closes[key] for key in sorted(session_closes.keys())]
    if not sorted_closes:
        return None, None, None
    close_ts, close = sorted_closes[-1]
    prev = sorted_closes[-2][1] if len(sorted_closes) >= 2 else None
    return close, close_ts, prev


def _combine_source(primary: str, secondary: str) -> str:
    states = {str(primary or "").lower(), str(secondary or "").lower()}
    if "live" in states:
        return "live"
    if "cache" in states:
        return "cache"
    return "none"


def _session_state_label(now_et: datetime) -> str:
    local = now_et.astimezone(_ET)
    if local.weekday() >= 5:
        return "CLOSED"
    t = local.time()
    if t < _RTH_OPEN:
        return "PRE_MARKET"
    if _RTH_OPEN <= t < _RTH_CLOSE:
        return "REGULAR"
    if t >= _RTH_CLOSE:
        return "AFTER_HOURS"
    return "CLOSED"


def _extended_label_for_session(*, session_state: str) -> str | None:
    normalized = str(session_state or "").strip().upper()
    if normalized == "PRE_MARKET":
        return "Pre-market"
    if normalized in {"AFTER_HOURS", "CLOSED"}:
        return "After-hours"
    return None


def _format_local_ts(value: datetime | None) -> str | None:
    if not isinstance(value, datetime):
        return None
    local = value.astimezone(_ET)
    return local.strftime("%Y-%m-%d %H:%M ET")
