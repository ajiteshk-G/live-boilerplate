"""MCP toolset manager: stdio, streamable-http, and SSE transports.

All servers share one ``AsyncExitStack`` so teardown is orderly. A server that
fails to start is logged and skipped -- one broken MCP server must never take
down the whole voice application.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from ..settings.schema import McpServerConfig
from .catalog import ToolCandidate, ToolInvoker

log = logging.getLogger(__name__)


def _flatten_content(result: Any) -> dict[str, Any]:
    """Turn an MCP CallToolResult into a JSON-serialisable payload the model can use."""
    if getattr(result, "isError", False):
        return {"error": _stringify(result)}

    texts: list[str] = []
    others: list[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)
        else:
            others.append(getattr(item, "type", type(item).__name__))

    structured = getattr(result, "structuredContent", None)
    payload: dict[str, Any] = {}
    if texts:
        payload["result"] = texts[0] if len(texts) == 1 else texts
    if structured:
        payload["structured"] = structured
    if others:
        payload["omitted_content_types"] = others
    if not payload:
        payload["result"] = ""
    return payload


def _stringify(result: Any) -> str:
    parts = [
        getattr(item, "text", None) or str(getattr(item, "type", item))
        for item in (getattr(result, "content", None) or [])
    ]
    return "; ".join(p for p in parts if p) or "unknown MCP error"


class McpToolsetManager:
    """Connects the configured MCP servers and exposes their tools as candidates."""

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._candidates: list[ToolCandidate] = []
        self.failures: dict[str, str] = {}

    @property
    def candidates(self) -> list[ToolCandidate]:
        return list(self._candidates)

    async def __aenter__(self) -> McpToolsetManager:
        await self._stack.__aenter__()
        for server in self._servers:
            if not server.enabled:
                continue
            try:
                await self._connect(server)
            except Exception as exc:
                self.failures[server.name] = str(exc)
                log.error(
                    "MCP server %r failed to start (%s); continuing without it",
                    server.name,
                    exc,
                )
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stack.aclose()

    async def _open_streams(self, server: McpServerConfig) -> tuple[Any, Any]:
        if server.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server.command or "",
                args=list(server.args),
                env={**os.environ, **server.env},
            )
            streams = await self._stack.enter_async_context(stdio_client(params))
        elif server.transport == "streamable_http":
            # The entry point was renamed between major versions of `mcp`:
            #   1.x -> streamablehttp_client
            #   2.x -> streamable_http_client
            # Accept either so this boilerplate is not pinned to one release.
            from mcp.client import streamable_http as _sh

            http_client = getattr(_sh, "streamable_http_client", None) or getattr(
                _sh, "streamablehttp_client", None
            )
            if http_client is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    "Could not find a streamable-http client in the installed `mcp` "
                    "package. Expected `streamable_http_client` (mcp 2.x) or "
                    "`streamablehttp_client` (mcp 1.x)."
                )

            sig = inspect.signature(http_client).parameters
            if "http_client" in sig:
                if server.headers:
                    import httpx

                    client = await self._stack.enter_async_context(
                        httpx.AsyncClient(headers=server.headers)
                    )
                    streams = await self._stack.enter_async_context(
                        http_client(server.url or "", http_client=client)
                    )
                else:
                    streams = await self._stack.enter_async_context(
                        http_client(server.url or "")
                    )
            else:
                streams = await self._stack.enter_async_context(
                    http_client(server.url or "", headers=server.headers or None)
                )
        elif server.transport == "sse":
            from mcp.client.sse import sse_client

            streams = await self._stack.enter_async_context(
                sse_client(server.url or "", headers=server.headers or None)
            )
        else:  # pragma: no cover - guarded by the Pydantic Literal
            raise ValueError(f"unknown transport {server.transport!r}")

        # streamablehttp_client yields a third element (a session-id callback).
        return streams[0], streams[1]

    async def _connect(self, server: McpServerConfig) -> None:
        from mcp import ClientSession

        read, write = await self._open_streams(server)
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=server.startup_timeout_s)

        self._sessions[server.name] = session
        self._locks[server.name] = asyncio.Lock()

        listed = await asyncio.wait_for(session.list_tools(), timeout=server.startup_timeout_s)
        added = 0
        for tool in listed.tools:
            if server.tools and tool.name not in server.tools:
                continue
            self._candidates.append(self._make_candidate(server.name, tool))
            added += 1
        log.info("MCP server %r: %d tool(s) available", server.name, added)

    def _make_candidate(self, server_name: str, tool: Any) -> ToolCandidate:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        return ToolCandidate(
            exposed_name=f"{server_name}__{tool.name}",
            description=getattr(tool, "description", "") or "",
            input_schema=dict(schema),
            invoke=self._make_invoker(server_name, tool.name),
            origin=f"mcp:{server_name}",
            raw_name=tool.name,
        )

    def _make_invoker(self, server_name: str, tool_name: str) -> ToolInvoker:
        async def invoke(args: dict[str, Any]) -> dict[str, Any]:
            session = self._sessions.get(server_name)
            if session is None:
                return {"error": f"MCP server {server_name!r} is not connected"}
            # Serialise per server: MCP sessions are not reliably safe for
            # concurrent calls, and the model can emit parallel function calls.
            async with self._locks[server_name]:
                result = await session.call_tool(tool_name, arguments=args or {})
            return _flatten_content(result)

        return invoke
