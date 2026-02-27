from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure repository root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.marketdata.chart_fetcher import range_mapping
from app.core.marketdata.query_graph import MarketQueryService
from app.core.marketdata.prewarm import (
    enqueue_prewarm_request,
    load_prewarm_config,
    save_prewarm_config,
    load_prewarm_status,
    next_due_at,
)
from app.core.env.load_env import load_env_from_dotenv_if_present
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import now_iso
from app.core.portfolio.portfolio_store import load_portfolio
from app.ui.components.brain import render_brain
from app.ui.components.horizon import render_horizon
from app.ui.components.pulse import render_pulse
from app.ui.components.topbar import render_topbar
from app.ui.theme import inject_global_css

st.set_page_config(page_title="Personal AI Stock Manager", layout="wide")
POLICY_PATH = Path(__file__).resolve().parents[1] / "core" / "drl" / "policies" / "drl_policy.yaml"
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_context_pack(
    ticker: str,
    interval: str,
    lookback_days: int,
    generate_hub_card: bool,
    hub_request_timeout_seconds: float | None,
    refresh_token: int,
    bedrock_region: str,
    bedrock_model_id: str,
    bedrock_claude_model_id: str,
    bedrock_openai_model_id: str,
) -> dict[str, Any]:
    provider = SampleMarketDataProvider()
    cache = DiskTTLCache(base_dir=".cache")

    bedrock_config = None
    if bedrock_region.strip() and (
        bedrock_model_id.strip() or bedrock_claude_model_id.strip() or bedrock_openai_model_id.strip()
    ):
        bedrock_config = {
            "region": bedrock_region.strip(),
            "model_id": bedrock_model_id.strip(),
            "claude_model_id": bedrock_claude_model_id.strip(),
            "openai_model_id": bedrock_openai_model_id.strip(),
        }

    context_pack = build_context_pack(
        ticker=ticker,
        now_iso=now_iso(),
        provider=provider,
        cache=cache,
        policy_path=str(POLICY_PATH),
        lookback_days=int(lookback_days),
        interval=interval,
        generate_hub_card=generate_hub_card,
        bedrock_config=bedrock_config,
        hub_request_timeout_seconds=hub_request_timeout_seconds,
    )
    context_pack.setdefault("meta", {})["latest_quote"] = _latest_quote_from_context_pack(
        ticker=ticker,
        context_pack=context_pack,
    )
    return context_pack


def main() -> None:
    load_env_from_dotenv_if_present()
    inject_global_css()
    st.markdown('<div class="app-title">AIStock Manager</div>', unsafe_allow_html=True)

    portfolio = load_portfolio()
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = portfolio.holdings[0].ticker if portfolio.holdings else "AAPL"
    if "ui_notices" not in st.session_state:
        st.session_state["ui_notices"] = []
    st.session_state.setdefault("pending_select_ticker", "")
    st.session_state.setdefault("topbar_input_error", "")
    st.session_state.setdefault("manual_refresh_requested", False)
    st.session_state.setdefault("manual_refresh_scope", "visible")
    st.session_state.setdefault("context_refresh_tokens", {})
    st.session_state.setdefault("prewarm_seen_completed_at", "")
    st.session_state.setdefault("prewarm_status", {})
    st.session_state.setdefault("ticker_click_refresh_at", {})
    _consume_card_selection_query_params()

    bedrock_region = os.getenv("AWS_REGION", "") or os.getenv("AWS_DEFAULT_REGION", "")
    bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "")
    bedrock_claude_model_id = os.getenv("BEDROCK_LLM_ID_CLAUDE", "")
    bedrock_openai_model_id = os.getenv("BEDROCK_LLM_ID_OPENAI", "")

    def context_loader(
        ticker: str,
        generate_hub_card: bool = False,
        interval: str = "1h",
        lookback_days: int = 60,
        hub_request_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _cached_context_pack(
                ticker=ticker.strip().upper(),
                interval=interval,
                lookback_days=int(lookback_days),
                generate_hub_card=generate_hub_card,
                hub_request_timeout_seconds=hub_request_timeout_seconds,
                refresh_token=_context_refresh_token(ticker=ticker),
                bedrock_region=bedrock_region,
                bedrock_model_id=bedrock_model_id,
                bedrock_claude_model_id=bedrock_claude_model_id,
                bedrock_openai_model_id=bedrock_openai_model_id,
            )
        except Exception:
            _add_notice("Some market data is temporarily unavailable. Showing degraded view.")
            return _degraded_context_pack(ticker=ticker, now=now_iso(), generate_hub_card=generate_hub_card)

    st.session_state["_context_loader"] = context_loader
    market_query = st.session_state.get("_market_query")
    if isinstance(market_query, MarketQueryService):
        market_query.context_loader = context_loader
    else:
        market_query = MarketQueryService(
            cache_dir=".cache/charts",
            context_loader=context_loader,
            short_interval="1h",
            short_lookback_days=60,
            long_interval="1h",
            long_lookback_days=60,
            cache_only=True,
        )
    st.session_state["_market_query"] = market_query

    _sync_prewarm_status()
    render_topbar()
    _render_configuration_panel()

    if bool(st.session_state.get("manual_refresh_requested", False)):
        refresh_scope = str(st.session_state.get("manual_refresh_scope", "visible")).strip().lower() or "visible"
        refresh_msg = _queue_manual_refresh(
            scope=refresh_scope,
            portfolio=portfolio,
            selected_ticker=st.session_state.get("selected_ticker"),
        )
        st.session_state["manual_refresh_requested"] = False
        _add_notice(refresh_msg)
        st.rerun()

    topbar_error = str(st.session_state.get("topbar_input_error", "")).strip()
    if topbar_error:
        st.warning(topbar_error)

    pending_select = _normalize_ticker(st.session_state.get("pending_select_ticker", ""))
    if pending_select:
        st.session_state["selected_ticker"] = pending_select
        st.session_state["pending_select_ticker"] = ""
        st.session_state["topbar_input_error"] = ""
        st.rerun()

    notices = list(dict.fromkeys(st.session_state.get("ui_notices", [])))
    for note in notices:
        st.info(note)

    portfolio = load_portfolio()
    col1, col2, col3 = st.columns([1.2, 2.25, 1.35], gap="medium")
    with col1:
        render_pulse(portfolio=portfolio, context_loader=context_loader, market_query=market_query)
    with col2:
        render_brain(
            selected_ticker=st.session_state.get("selected_ticker"),
            context_loader=context_loader,
            policy_path=str(POLICY_PATH),
            market_query=market_query,
        )
    with col3:
        render_horizon(context_loader=context_loader, market_query=market_query)


def _add_notice(message: str) -> None:
    notices = st.session_state.setdefault("ui_notices", [])
    notices.append(message)


def _sync_prewarm_status() -> None:
    status = load_prewarm_status()
    st.session_state["prewarm_status"] = status
    completed_at = str(status.get("last_completed_at", "")).strip()
    if not completed_at:
        return
    seen = str(st.session_state.get("prewarm_seen_completed_at", "")).strip()
    if completed_at == seen:
        return
    touched = {
        str(item).strip().upper()
        for item in list(status.get("last_tickers", []) or [])
        if str(item).strip()
    }
    _clear_market_query_cache()
    _bump_context_refresh_tokens(tickers=touched)
    _clear_horizon_session_cache()
    if seen:
        attempted = int(status.get("attempted", 0) or 0)
        live = int(status.get("live", 0) or 0)
        _add_notice(f"Background refresh complete: attempted={attempted}, live={live}.")
    st.session_state["prewarm_seen_completed_at"] = completed_at


def _render_configuration_panel() -> None:
    with st.expander("Configuration", expanded=False):
        st.markdown("**Prewarm Scheduler**")
        cfg = load_prewarm_config()
        status = st.session_state.get("prewarm_status", {})
        with st.form("prewarm_config_form"):
            enabled = st.checkbox("Enable background prewarm", value=bool(cfg.get("enabled", True)))
            market_minutes = st.number_input(
                "Trading hours cadence (minutes, 09:30-16:00 ET)",
                min_value=5,
                max_value=240,
                step=5,
                value=int(cfg.get("market_minutes", 30)),
            )
            after_hours_minutes = st.number_input(
                "After-hours cadence (minutes, 16:00-20:00 ET)",
                min_value=5,
                max_value=360,
                step=5,
                value=int(cfg.get("after_hours_minutes", 60)),
            )
            off_hours_minutes = st.number_input(
                "Off-hours cadence (minutes, 20:00-09:30 ET)",
                min_value=60,
                max_value=1440,
                step=60,
                value=int(cfg.get("off_hours_minutes", 720)),
            )
            weekend_minutes = st.number_input(
                "Weekend cadence (minutes)",
                min_value=60,
                max_value=1440,
                step=60,
                value=int(cfg.get("weekend_minutes", 720)),
            )
            cache_max_age_hours = st.number_input(
                "Cache max age (hours)",
                min_value=1,
                max_value=168,
                step=1,
                value=int(cfg.get("cache_max_age_hours", 48)),
            )
            cache_budget_mb = st.number_input(
                "Cache budget (MB)",
                min_value=50,
                max_value=4096,
                step=50,
                value=int(cfg.get("cache_budget_mb", 250)),
            )
            horizon_enabled = st.checkbox("Include Horizon universe in prewarm", value=bool(cfg.get("horizon_enabled", True)))
            submitted = st.form_submit_button("Save Prewarm Settings", width="stretch")
        if submitted:
            save_prewarm_config(
                {
                    "enabled": enabled,
                    "market_minutes": int(market_minutes),
                    "after_hours_minutes": int(after_hours_minutes),
                    "off_hours_minutes": int(off_hours_minutes),
                    "weekend_minutes": int(weekend_minutes),
                    "cache_max_age_hours": int(cache_max_age_hours),
                    "cache_budget_mb": int(cache_budget_mb),
                    "horizon_enabled": bool(horizon_enabled),
                }
            )
            _add_notice("Prewarm configuration saved.")
            st.rerun()
        next_due = str(status.get("next_due_at", "")).strip()
        if not next_due:
            next_due = next_due_at(config=cfg, status=status)
        st.caption(
            "Worker status: "
            f"last={status.get('last_completed_at', 'n/a')} | "
            f"next={next_due} | "
            f"attempted={status.get('attempted', 0)} | "
            f"live={status.get('live', 0)} | cache={status.get('cache', 0)}"
        )


def _context_refresh_token(ticker: str) -> int:
    symbol = str(ticker or "").strip().upper()
    tokens = st.session_state.get("context_refresh_tokens")
    if not isinstance(tokens, dict):
        return 0
    value = tokens.get(symbol, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _consume_card_selection_query_params() -> bool:
    selected: str | None = None
    for key in ("select_ticker", "pulse_select", "horizon_select"):
        candidate = _normalize_ticker(st.query_params.get(key, ""))
        if candidate and selected is None:
            selected = candidate
        try:
            st.query_params.pop(key)
        except Exception:
            continue
    if selected:
        st.session_state["selected_ticker"] = selected
        st.session_state["pending_select_ticker"] = ""
        st.session_state["topbar_input_error"] = ""
        _queue_ticker_click_refresh(selected)
        return True
    return False


def _clear_market_query_cache() -> None:
    query = st.session_state.get("_market_query")
    if isinstance(query, MarketQueryService):
        try:
            query.clear_local_cache()
        except Exception:
            return


def _queue_ticker_click_refresh(ticker: str) -> None:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return
    throttle = st.session_state.setdefault("ticker_click_refresh_at", {})
    if not isinstance(throttle, dict):
        throttle = {}
        st.session_state["ticker_click_refresh_at"] = throttle

    now_ts = datetime.now().timestamp()
    last_ts_raw = throttle.get(symbol, 0.0)
    try:
        last_ts = float(last_ts_raw)
    except (TypeError, ValueError):
        last_ts = 0.0
    if (now_ts - last_ts) < 45.0:
        return

    enqueue_prewarm_request(
        scope="brain",
        tickers={symbol},
        range_keys=_refresh_scope_ranges(scope="brain", selected_ticker=symbol),
        reason="ticker_click",
        requested_by="ui",
    )
    throttle[symbol] = now_ts


def _queue_manual_refresh(scope: str, portfolio: Any, selected_ticker: str | None) -> str:
    scope_norm = str(scope or "visible").strip().lower()
    if scope_norm not in {"visible", "brain", "horizon", "all"}:
        scope_norm = "visible"

    tickers = _refresh_scope_tickers(scope=scope_norm, portfolio=portfolio, selected_ticker=selected_ticker)
    ranges = _refresh_scope_ranges(scope=scope_norm, selected_ticker=selected_ticker)
    request = enqueue_prewarm_request(
        scope=scope_norm,
        tickers=tickers,
        range_keys=tuple(ranges),
        reason="manual_refresh",
        requested_by="ui",
    )
    st.session_state["last_refresh_report"] = {
        "scope": scope_norm,
        "tickers": len(tickers),
        "ranges": list(ranges),
        "attempted": 0,
        "live": 0,
        "cache": 0,
        "none": 0,
        "errors": 0,
        "queued": True,
        "request_id": request.get("request_id", ""),
        "cache_hygiene": {"scanned": 0, "removed": 0},
    }
    scope_label = {
        "visible": "visible cards",
        "brain": "Brain",
        "horizon": "Horizon",
        "all": "all scopes",
    }.get(scope_norm, "selected scope")
    return f"Refresh queued for {scope_label}: {len(tickers)} tickers, ranges={','.join(ranges)}."


def _refresh_scope_tickers(scope: str, portfolio: Any, selected_ticker: str | None) -> set[str]:
    scope_norm = str(scope or "visible").strip().lower()
    holdings = {
        str(getattr(item, "ticker", "")).strip().upper()
        for item in list(getattr(portfolio, "holdings", []) or [])
        if str(getattr(item, "ticker", "")).strip()
    }
    selected = str(selected_ticker or "").strip().upper()
    if scope_norm == "brain":
        return {selected} if selected else set()
    if scope_norm == "horizon":
        return _load_sp100_universe()

    # visible = Pulse holdings + active Brain ticker
    if selected:
        holdings.add(selected)
    return holdings


def _bump_context_refresh_tokens(tickers: set[str]) -> None:
    tokens = st.session_state.setdefault("context_refresh_tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}
        st.session_state["context_refresh_tokens"] = tokens
    for ticker in tickers:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        current = tokens.get(symbol, 0)
        try:
            tokens[symbol] = int(current) + 1
        except (TypeError, ValueError):
            tokens[symbol] = 1


def _refresh_scope_ranges(scope: str, selected_ticker: str | None) -> tuple[str, ...]:
    scope_norm = str(scope or "visible").strip().lower()
    if scope_norm == "all":
        return ("1D", "1W", "1M", "3M", "YTD", "1Y")
    if scope_norm == "horizon":
        # Ranking requires 1W only; 1D sparklines are fetched only for the final top-10.
        return ("1W",)
    if scope_norm == "brain":
        return ("1D", "1W", "1M", "3M", "YTD", "1Y")

    ranges: list[str] = ["1D", "1W"]
    return tuple(ranges)


def _purge_chart_cache_for_tickers(tickers: set[str], range_keys: set[str] | None = None) -> int:
    if not tickers:
        return 0
    cache = DiskTTLCache(base_dir=".cache/charts")
    removed = 0
    ranges = sorted(range_keys) if range_keys else ["1D", "1W", "1M", "3M", "YTD", "1Y"]
    for ticker in tickers:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        for range_key in ranges:
            mapping = range_mapping(range_key)
            cache_key = (
                f"brain-chart:{symbol}:{range_key}:{mapping['period']}:"
                f"{mapping['interval']}:{int(bool(mapping['prepost']))}"
            )
            path = Path(cache.path_for_key(cache_key))
            if path.exists():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
    return removed


def _clear_horizon_session_cache() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("horizon_movers_cache_"):
            st.session_state.pop(key, None)


def _cache_hygiene_cleanup(base_dirs: list[Path]) -> dict[str, int]:
    scanned = 0
    removed = 0
    for base_dir in base_dirs:
        path = Path(base_dir)
        if not path.exists():
            continue
        for file_path in path.rglob("*.json"):
            if not file_path.is_file():
                continue
            scanned += 1
            try:
                if file_path.stat().st_size == 0:
                    file_path.unlink()
                    removed += 1
                    continue
            except OSError:
                continue
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                try:
                    file_path.unlink()
                    removed += 1
                except OSError:
                    pass
                continue
            if not isinstance(payload, dict):
                try:
                    file_path.unlink()
                    removed += 1
                except OSError:
                    pass
                continue
    return {"scanned": scanned, "removed": removed}


def _load_sp100_universe() -> set[str]:
    path = Path("app/data/sp100_universe.json")
    if not path.exists():
        return {
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA",
            "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CHTR", "CL",
            "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "D", "DHR",
            "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG",
            "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "JNJ", "JPM", "KHC",
            "KMI", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDT", "META",
            "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA",
            "ORCL", "OXY", "PANW", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
            "SBUX", "SCHW", "SLB", "SO", "SPG", "T", "TGT", "TMO", "TXN", "UNH",
            "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "AVGO",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(payload, list):
        return {
            str(item).strip().upper()
            for item in payload
            if str(item).strip()
        }
    if isinstance(payload, dict):
        tickers = payload.get("tickers", [])
        if isinstance(tickers, list):
            return {
                str(item).strip().upper()
                for item in tickers
                if str(item).strip()
            }
    return set()


def _latest_quote_from_context_pack(ticker: str, context_pack: dict[str, Any]) -> dict[str, Any]:
    bars = context_pack.get("prices", {}).get("bars", [])
    close_price = None
    prev_close = None
    close_ts = None
    if isinstance(bars, list) and bars:
        last = bars[-1] if isinstance(bars[-1], dict) else {}
        try:
            close_price = float(last.get("close"))
        except Exception:
            close_price = None
        close_ts = str(last.get("ts")) if last.get("ts") else None
        if len(bars) >= 2 and isinstance(bars[-2], dict):
            try:
                prev_close = float(bars[-2].get("close"))
            except Exception:
                prev_close = None

    today_abs = None
    today_pct = None
    if isinstance(close_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close not in {0.0, 0}:
        today_abs = float(close_price) - float(prev_close)
        today_pct = (today_abs / float(prev_close)) * 100.0

    return {
        "symbol": ticker.strip().upper(),
        "currency": "USD",
        "close_price": close_price,
        "close_ts": close_ts,
        "close_ts_local": None,
        "prev_close_price": prev_close,
        "last_regular": close_price,
        "last_regular_ts": close_ts,
        "after_hours_price": None,
        "after_hours_ts": None,
        "after_hours_ts_local": None,
        "latest_price": close_price,
        "latest_ts": close_ts,
        "latest_ts_local": None,
        "latest_source": "close",
        "today_change_abs": today_abs,
        "today_change_pct": today_pct,
        "after_hours_change_abs": None,
        "after_hours_change_pct": None,
        "source": "context-pack",
        "quality_flags": [],
        "error": None,
    }


def _degraded_context_pack(ticker: str, now: str, generate_hub_card: bool) -> dict[str, Any]:
    context_pack: dict[str, Any] = {
        "meta": {
            "ticker": ticker.strip().upper(),
            "generated_at": now,
            "interval": "1h",
            "lookback_days": 60,
            "data_quality": {
                "prices": {
                    "as_of": now,
                    "now": now,
                    "age_minutes": 0.0,
                    "stale": True,
                    "stale_minutes_threshold": 90,
                },
                "indicators": {
                    "as_of": now,
                    "now": now,
                    "age_minutes": 0.0,
                    "stale": True,
                    "stale_minutes_threshold": 90,
                },
                "overall_stale": True,
                "notes": ["DEGRADED_UI: context generation unavailable", "TOOL_DOWN"],
            },
        },
        "prices": {"as_of": now, "bars": []},
        "indicators": {"as_of": now, "metrics": {}},
        "drl": {
            "result": {
                "regime_1D": "NEUTRAL",
                "regime_1W": "NEUTRAL",
                "action_final": "WAIT",
                "confidence_cap": 0,
                "gates_triggered": [],
                "conflicts": ["TOOL_DOWN"],
                "decision_trace": {
                    "policy_id": "",
                    "policy_version": "",
                    "ticker": ticker.strip().upper(),
                    "profile": "",
                    "timestamp": now,
                    "score_final": 0.0,
                    "base_action": "WAIT",
                    "action_final": "WAIT",
                },
            },
            "decision_trace": {
                "policy_id": "",
                "policy_version": "",
                "ticker": ticker.strip().upper(),
                "profile": "",
                "timestamp": now,
                "score_final": 0.0,
                "base_action": "WAIT",
                "action_final": "WAIT",
            },
        },
    }

    if generate_hub_card:
        context_pack["hub_card"] = {
            "meta": {
                "ticker": ticker.strip().upper(),
                "generated_at": now,
                "policy_id": "",
                "policy_version": "",
                "profile": "",
                "mode": "DEGRADED",
            },
            "summary": {
                "action_final": "WAIT",
                "confidence_cap": 0,
                "one_liner": "Data services are temporarily unavailable; DRL view is degraded and held at WAIT.",
            },
            "drivers": [
                {"text": "System-level data degradation is active for this ticker view.", "citations": ["indicator:price_last"]},
                {"text": "Trust state is reduced until fresh market data is available.", "citations": ["indicator:ema_50"]},
            ],
            "conflicts": [
                {"text": "Tool availability is reduced, so external evidence is unavailable.", "citations": ["indicator:rsi_14"]}
            ],
            "watch": [
                {"text": "Watch for successful data refresh before relying on directional cues.", "citations": ["indicator:macd"]},
                {"text": "Watch trust badges for stale and tool-down resolution.", "citations": ["indicator:atr_pct"]},
            ],
            "evidence": {
                "used_ids": [
                    "indicator:price_last",
                    "indicator:ema_50",
                    "indicator:rsi_14",
                    "indicator:macd",
                    "indicator:atr_pct",
                ]
            },
        }

    return context_pack


def _normalize_ticker(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if not _TICKER_PATTERN.fullmatch(text):
        return None
    return text


if __name__ == "__main__":
    main()
