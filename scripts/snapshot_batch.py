from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.marketdata.yfinance_provider import SampleMarketDataProvider
from app.core.orchestration.cache import DiskTTLCache
from app.core.orchestration.context_pack import build_context_pack
from app.core.orchestration.time_utils import now_iso
from app.core.portfolio.portfolio_store import load_portfolio
from app.core.replay.artifact_store import save_artifact

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context packs and save replay snapshots for a batch of tickers")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--include-hub", action="store_true", default=False)
    args = parser.parse_args()

    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"

    tickers = _resolve_tickers(args.tickers)
    if not tickers:
        tickers = DEFAULT_UNIVERSE

    provider = SampleMarketDataProvider()
    cache = DiskTTLCache(base_dir=".cache")
    run_now = now_iso()

    saved_paths: list[str] = []
    for ticker in tickers:
        try:
            pack = build_context_pack(
                ticker=ticker,
                now_iso=run_now,
                provider=provider,
                cache=cache,
                policy_path=str(policy_path),
                lookback_days=args.lookback_days,
                interval=args.interval,
                generate_hub_card=bool(args.include_hub),
                bedrock_config=None,
            )
            path = save_artifact(
                ticker=ticker,
                policy_path=str(policy_path),
                context_pack=pack,
                now_iso=run_now,
                notes=["snapshot_batch"],
            )
            saved_paths.append(path)
            action = pack.get("drl", {}).get("result", {}).get("action_final", "WAIT")
            conf = pack.get("drl", {}).get("result", {}).get("confidence_cap", 0)
            print(f"{ticker} | action={action} | conf={conf} | saved={path}")
        except Exception as exc:
            print(f"{ticker} | ERROR | {exc}")

    print("---")
    print(f"saved_count={len(saved_paths)}")
    print("paths=")
    for path in saved_paths:
        print(path)


def _resolve_tickers(arg_tickers: list[str] | None) -> list[str]:
    if arg_tickers:
        return _dedupe([t.strip().upper() for t in arg_tickers if t.strip()])

    portfolio_tickers = [holding.ticker for holding in load_portfolio().holdings]
    movers = _top_movers_from_universe(limit=6)
    return _dedupe(portfolio_tickers + movers)


def _top_movers_from_universe(limit: int) -> list[str]:
    universe = _load_universe()
    sample = universe[:20]
    provider = SampleMarketDataProvider()
    cache = DiskTTLCache(base_dir=".cache")
    policy_path = Path(__file__).resolve().parents[1] / "app" / "core" / "drl" / "policies" / "drl_policy.yaml"
    run_now = now_iso()

    scored: list[tuple[str, float]] = []
    for ticker in sample:
        try:
            pack = build_context_pack(
                ticker=ticker,
                now_iso=run_now,
                provider=provider,
                cache=cache,
                policy_path=str(policy_path),
                lookback_days=60,
                interval="1h",
                generate_hub_card=False,
            )
            move = _move_percent(pack)
            scored.append((ticker, move))
        except Exception:
            continue

    scored = sorted(scored, key=lambda item: (-abs(item[1]), item[0]))
    return [ticker for ticker, _ in scored[:limit]]


def _load_universe() -> list[str]:
    path = Path("app/data/sp500_universe.json")
    if not path.exists():
        return DEFAULT_UNIVERSE

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_UNIVERSE

    if isinstance(data, list):
        vals = [str(x).strip().upper() for x in data if str(x).strip()]
        return vals or DEFAULT_UNIVERSE

    if isinstance(data, dict):
        raw = data.get("tickers", [])
        if isinstance(raw, list):
            vals = [str(x).strip().upper() for x in raw if str(x).strip()]
            return vals or DEFAULT_UNIVERSE

    return DEFAULT_UNIVERSE


def _move_percent(context_pack: dict) -> float:
    bars = context_pack.get("prices", {}).get("bars", [])
    if len(bars) < 2:
        return 0.0

    first_close = float(bars[0].get("close", 0.0))
    last_close = float(bars[-1].get("close", 0.0))
    if first_close == 0.0:
        return 0.0
    return ((last_close - first_close) / first_close) * 100.0


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


if __name__ == "__main__":
    main()
