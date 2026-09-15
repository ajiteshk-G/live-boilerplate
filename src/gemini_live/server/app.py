"""FastAPI application: serves the web client and relays audio to the Live API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..live.client import build_client
from ..live.runner import LiveSessionRunner
from ..pipeline import ToolPipeline, ToolPipelineResult
from ..settings.capabilities import filter_for_model
from ..settings.schema import AppConfig, redact
from .ws_protocol import decode_client_message, encode_event

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class _WebSocketSink:
    """Bridges runner events onto a WebSocket."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._open = True

    async def audio(self, pcm: bytes) -> None:
        if self._open:
            with contextlib.suppress(Exception):
                await self._ws.send_bytes(pcm)

    async def event(self, kind: str, payload: dict[str, Any]) -> None:
        if self._open:
            with contextlib.suppress(Exception):
                await self._ws.send_text(encode_event(kind, payload))

    def close(self) -> None:
        self._open = False


def create_app(cfg: AppConfig) -> FastAPI:
    cfg, warnings = filter_for_model(cfg)
    for warning in warnings:
        log.warning("%s", warning)

    app = FastAPI(title=cfg.app.name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"tools": None, "client": None}

    @app.on_event("startup")
    async def _startup() -> None:
        # MCP servers and tool curation run once, shared across connections.
        client = build_client(cfg)
        pipeline = ToolPipeline(cfg, client)
        await pipeline.__aenter__()
        result = await pipeline.build()
        state["client"] = client
        state["pipeline"] = pipeline
        state["tools"] = result
        log.info(
            "ready: %d tool(s) selected costing %d tokens per turn",
            len(result.selection.selected),
            result.selection.total_tokens,
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        pipeline = state.get("pipeline")
        if pipeline is not None:
            with contextlib.suppress(Exception):
                await pipeline.__aexit__(None, None, None)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        tools: ToolPipelineResult | None = state.get("tools")
        return {
            "ok": True,
            "model": cfg.model.name,
            "tools": len(tools.selection.selected) if tools else 0,
        }

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        tools: ToolPipelineResult | None = state.get("tools")
        summary = {
            "config": redact(cfg),
            "warnings": warnings,
            "tools": {
                "selected": [
                    {
                        "name": c.exposed_name,
                        "origin": c.origin,
                        "tokens": c.token_cost,
                        "score": round(c.score, 4),
                        "pinned": c.pinned,
                    }
                    for c in tools.selection.selected
                ]
                if tools
                else [],
                "catalog_size": tools.selection.catalog_size if tools else 0,
                "tokens_per_turn": tools.selection.total_tokens if tools else 0,
                "tokens_saved": tools.selection.saved_tokens if tools else 0,
                "mcp_failures": tools.mcp_failures if tools else {},
            },
            "domains": {
                "allow": cfg.search.domains.allow,
                "deny": cfg.search.domains.deny,
                "enforcement": cfg.search.enforcement,
                "google_search": cfg.search.google_search.enabled,
            },
        }
        return JSONResponse(summary)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        tools: ToolPipelineResult | None = state.get("tools")
        if tools is None:
            await ws.send_text(encode_event("error", {"message": "server still starting"}))
            await ws.close()
            return

        sink = _WebSocketSink(ws)
        runner = LiveSessionRunner(cfg, state["client"], tools.registry, sink=sink)
        task = asyncio.create_task(runner.run())
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    await runner.uplink.audio(data)
                elif (text := message.get("text")) is not None:
                    payload = decode_client_message(text)
                    kind = payload.get("type")
                    if kind == "text" and payload.get("text"):
                        await runner.uplink.text(str(payload["text"]))
                    elif kind == "mic" and payload.get("on") is False:
                        await runner.uplink.mic_off()
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket relay failed")
        finally:
            sink.close()
            await runner.close()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
