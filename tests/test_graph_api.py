from __future__ import annotations

from datetime import datetime, timezone

from starlette.testclient import TestClient

from app.api.graph_api import create_app
from app.core.marketdata.chart_fetcher import Bar, ChartSeries


class FakeQueryService:
    def __init__(self) -> None:
        self.revalidate_calls: list[tuple[set[str], tuple[str, ...]]] = []
        self.long_calls: list[tuple[str, bool]] = []

    def revalidate_tickers(self, *, tickers: set[str], range_keys: tuple[str, ...]) -> dict[str, int]:
        self.revalidate_calls.append((set(tickers), tuple(range_keys)))
        return {"attempted": 1, "live": 1, "cache": 0, "none": 0, "errors": 0}

    def pulse_card_data(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "context_pack": {"meta": {"ticker": ticker}, "drl": {"result": {"action_final": "WAIT"}}},
            "quote": {"latest_price": 123.45, "source": "live"},
            "series_1d": ChartSeries(
                bars=[
                    Bar(
                        ts=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
                        open=120.0,
                        high=124.0,
                        low=119.0,
                        close=123.45,
                        volume=1000.0,
                    )
                ],
                as_of=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
                source="live",
                error=None,
                quality_flags=set(),
                cache_path=".cache/charts/test.json",
                cache_age_minutes=0.0,
                cache_hit=False,
                stale_cache=False,
                attempts=1,
            ),
        }

    def brain_card_data(self, ticker: str, *, generate_hub_card: bool) -> dict:
        self.long_calls.append((ticker, bool(generate_hub_card)))
        payload = self.pulse_card_data(ticker=ticker)
        payload["context_pack"]["meta"]["hub_requested"] = bool(generate_hub_card)
        return payload


def test_query_short_success(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "1")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    resp = client.post("/api/query/short", json={"ticker": "goog"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["query"] == "short"
    assert body["ticker"] == "GOOG"
    assert body["data"]["quote"]["latest_price"] == 123.45
    assert body["data"]["series_1d"]["source"] == "live"


def test_query_short_revalidate_calls_delta_refresh(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "1")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    resp = client.post("/api/query/short", json={"ticker": "NVDA", "revalidate": True})
    assert resp.status_code == 200
    assert fake.revalidate_calls
    assert fake.revalidate_calls[0][0] == {"NVDA"}
    assert fake.revalidate_calls[0][1] == ("1D", "1W")


def test_query_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", raising=False)
    monkeypatch.setenv("GRAPH_API_KEY", "secret-key")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    denied = client.post("/api/query/short", json={"ticker": "AAPL"})
    assert denied.status_code == 401

    allowed = client.post(
        "/api/query/short",
        json={"ticker": "AAPL"},
        headers={"x-api-key": "secret-key"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["ok"] is True


def test_query_long_passes_generate_hub_flag(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "1")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    resp = client.post("/api/query/long", json={"ticker": "MSFT", "generate_hub_card": True})
    assert resp.status_code == 200
    assert fake.long_calls == [("MSFT", True)]
    assert resp.json()["generate_hub_card"] is True


def test_query_rejects_invalid_ticker(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "1")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    resp = client.post("/api/query/short", json={"ticker": "$BAD"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "INVALID_TICKER"


def test_query_fails_closed_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.delenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", raising=False)
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    resp = client.post("/api/query/short", json={"ticker": "AAPL"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "AUTH_NOT_CONFIGURED"


def test_revalidate_is_throttled(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("GRAPH_API_ALLOW_UNAUTH_LOCALHOST", "1")
    fake = FakeQueryService()
    app = create_app(service_factory=lambda: fake)
    client = TestClient(app)

    first = client.post("/api/query/short", json={"ticker": "AAPL", "revalidate": True})
    assert first.status_code == 200
    second = client.post("/api/query/short", json={"ticker": "AAPL", "revalidate": True})
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "REVALIDATE_THROTTLED"
