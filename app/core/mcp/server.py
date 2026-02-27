from __future__ import annotations

import json
import sys
from typing import Annotated, Any, Callable

from pydantic import Field

from app.core.mcp.tools import events, macro, news, prices

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    FastMCP = None  # type: ignore[assignment]

SERVER_NAME = "personal-ai-stock-manager"
SERVER_DESCRIPTION = "Read-only stock context tools: prices, news, macro, events"


def get_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": prices.TOOL_NAME,
            "description": prices.TOOL_DESCRIPTION,
            "input_schema": prices.ARGUMENTS_SCHEMA,
            "handler": prices.get_prices,
        },
        {
            "name": news.TOOL_NAME,
            "description": news.TOOL_DESCRIPTION,
            "input_schema": news.ARGUMENTS_SCHEMA,
            "handler": news.get_news,
        },
        {
            "name": macro.TOOL_NAME,
            "description": macro.TOOL_DESCRIPTION,
            "input_schema": macro.ARGUMENTS_SCHEMA,
            "handler": macro.get_macro_snapshot,
        },
        {
            "name": events.TOOL_NAME,
            "description": events.TOOL_DESCRIPTION,
            "input_schema": events.ARGUMENTS_SCHEMA,
            "handler": events.get_events,
        },
    ]


def create_mcp_server() -> Any:
    if FastMCP is None:
        raise RuntimeError("MCP SDK not installed. Install dependency: mcp>=1.10.0")

    mcp = FastMCP(name=SERVER_NAME, description=SERVER_DESCRIPTION)

    def prices_tool(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. AAPL")],
        interval: Annotated[str, Field(description="Bar interval, e.g. 1h, 1d")] = "1h",
        lookback_days: Annotated[int, Field(ge=1, le=365, description="Lookback days")] = 60,
    ) -> dict[str, Any]:
        return prices.get_prices(ticker=ticker, interval=interval, lookback_days=lookback_days)

    def news_tool(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. AAPL")],
        lookback_hours: Annotated[int, Field(ge=1, le=168, description="News lookback in hours")] = 48,
        max_items: Annotated[int, Field(ge=1, le=30, description="Maximum number of items")] = 12,
    ) -> dict[str, Any]:
        return news.get_news(ticker=ticker, lookback_hours=lookback_hours, max_items=max_items)

    def macro_tool(
        lookback_days: Annotated[int, Field(ge=1, le=30, description="Lookback days for macro snapshot")] = 1,
    ) -> dict[str, Any]:
        return macro.get_macro_snapshot(lookback_days=lookback_days)

    def events_tool(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. AAPL")],
        max_items: Annotated[int, Field(ge=0, le=20, description="Maximum number of event items")] = 10,
    ) -> dict[str, Any]:
        return events.get_events(ticker=ticker, max_items=max_items)

    _register_tool(mcp, prices_tool, prices.TOOL_NAME, prices.TOOL_DESCRIPTION)
    _register_tool(mcp, news_tool, news.TOOL_NAME, news.TOOL_DESCRIPTION)
    _register_tool(mcp, macro_tool, macro.TOOL_NAME, macro.TOOL_DESCRIPTION)
    _register_tool(mcp, events_tool, events.TOOL_NAME, events.TOOL_DESCRIPTION)

    return mcp


def _register_tool(mcp: Any, handler: Callable[..., Any], name: str, description: str) -> None:
    try:
        mcp.tool(name=name, description=description)(handler)
    except TypeError:
        # Older SDK fallback if named metadata args are unavailable.
        mcp.tool()(handler)


class _FallbackStdioServer:
    """Minimal stdio JSON-RPC fallback if MCP SDK is unavailable."""

    def __init__(self) -> None:
        self._tools = {td["name"]: td for td in get_tool_definitions()}
        self._running = True

    def run(self) -> None:
        while self._running:
            raw = sys.stdin.readline()
            if raw == "":
                break
            raw = raw.strip()
            if not raw:
                continue

            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(request, dict):
                continue

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            if method == "initialize":
                self._send_result(
                    request_id,
                    {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
                    },
                )
                continue

            if method == "notifications/initialized":
                continue

            if method == "tools/list":
                tools = [
                    {
                        "name": td["name"],
                        "description": td["description"],
                        "inputSchema": td["input_schema"],
                    }
                    for td in get_tool_definitions()
                ]
                self._send_result(request_id, {"tools": tools})
                continue

            if method == "tools/call":
                self._handle_tool_call(request_id, params)
                continue

            if method == "ping":
                self._send_result(request_id, {})
                continue

            if method == "shutdown":
                self._send_result(request_id, {})
                continue

            if method == "exit":
                self._running = False
                continue

            self._send_error(request_id, code=-32601, message=f"Method not found: {method}")

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> None:
        try:
            tool_name = params.get("name")
            args = params.get("arguments") or {}
            tool_def = self._tools.get(tool_name)
            if tool_def is None:
                self._send_error(request_id, code=-32602, message=f"Unknown tool: {tool_name}")
                return

            handler = tool_def["handler"]
            result = handler(**args)
            self._send_result(
                request_id,
                {
                    "structuredContent": result,
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=True)}],
                    "isError": bool(isinstance(result, dict) and "error" in result),
                },
            )
        except Exception as exc:
            self._send_error(request_id, code=-32000, message="Tool call failed", data={"reason": str(exc)})

    def _send_result(self, request_id: Any, result: dict[str, Any]) -> None:
        if request_id is None:
            return
        payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    def _send_error(self, request_id: Any, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        if request_id is None:
            return
        err: dict[str, Any] = {"code": code, "message": message}
        if data:
            err["data"] = data
        payload = {"jsonrpc": "2.0", "id": request_id, "error": err}
        sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
        sys.stdout.flush()


def main() -> None:
    if FastMCP is not None:
        server = create_mcp_server()
        server.run(transport="stdio")
        return

    fallback = _FallbackStdioServer()
    fallback.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err = {
            "error": {
                "code": "TOOL_DOWN",
                "message": "MCP server failed to start",
                "details": {"reason": str(exc)},
            }
        }
        sys.stderr.write(json.dumps(err, ensure_ascii=True) + "\n")
        raise SystemExit(1)
