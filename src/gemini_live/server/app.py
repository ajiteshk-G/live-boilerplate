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
from ..live.runner import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, LiveSessionRunner
from ..pipeline import ToolPipeline, ToolPipelineResult
from ..settings.capabilities import filter_for_model
from ..settings.schema import AppConfig, redact
from ..settings.voices import LIVE_VOICES, VOICE_GENDERS
from ..telemetry import activate_telemetry, get_telemetry_snapshot
from .ws_protocol import apply_client_message, decode_client_message, encode_event

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

_HANGUP_FLUSH_SECONDS = 5.0
"""How long to let the runner finish its last turn after the user hangs up."""


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

    # Activate gemini-live-telemetry before any google-genai client/session is opened
    telemetry_status = activate_telemetry(cfg)

    app = FastAPI(title=cfg.app.name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"tools": None, "client": None, "telemetry": telemetry_status}

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
            "telemetry_enabled": cfg.telemetry.enabled,
        }

    @app.get("/api/telemetry")
    async def api_telemetry() -> JSONResponse:
        return JSONResponse(get_telemetry_snapshot(cfg))

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        tools: ToolPipelineResult | None = state.get("tools")
        summary = {
            "config": redact(cfg),
            "warnings": warnings,
            "telemetry": get_telemetry_snapshot(cfg),
            "voices": [
                {
                    "name": name,
                    "gender": VOICE_GENDERS.get(name, "female"),
                    "character": character,
                }
                for name, character in LIVE_VOICES.items()
            ],
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
                "google_search": cfg.search.google_search.enabled,
            },
            # The wire format is fixed by the Live API. Serving it keeps the
            # browser from keeping its own copy that can drift.
            "audio": {
                "input_sample_rate": INPUT_SAMPLE_RATE,
                "output_sample_rate": OUTPUT_SAMPLE_RATE,
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
        # Apply per-session locale, voice, and topic overrides requested by the UI.
        session_cfg = cfg.for_session_params(
            locale=ws.query_params.get("locale"),
            voice=ws.query_params.get("voice"),
            topic=ws.query_params.get("topic"),
        )
        if session_cfg is not cfg:
            log.info(
                "session params: voice=%s (%s, %s), language=%s, topic=%r",
                session_cfg.speech.voice_name,
                session_cfg.agent.name,
                session_cfg.agent.gender,
                session_cfg.speech.language_code,
                session_cfg.model.talk_only_about,
            )
        # Build a fresh client per session so a long-running server picks up
        # refreshed Application Default Credentials rather than holding an
        # expired startup token.
        runner = LiveSessionRunner(
            session_cfg, build_client(session_cfg), tools.registry, sink=sink
        )
        task = asyncio.create_task(runner.run())

        async def _relay_client() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    await runner.uplink.audio(data)
                elif (text := message.get("text")) is not None:
                    payload = decode_client_message(text)
                    if not await apply_client_message(payload, runner):
                        # Acknowledge first: once runner.run() returns, the
                        # endpoint cancels this relay, so an ack sent after the
                        # wait below would race with that cancellation.
                        await sink.event(
                            "session_ended", {"reason": "ended by the user"}
                        )
                        # Then give the runner a moment to commit its final turn
                        # and emit the closing usage events.
                        await asyncio.wait({task}, timeout=_HANGUP_FLUSH_SECONDS)
                        break

        relay_task = asyncio.create_task(_relay_client())
        try:
            done, pending = await asyncio.wait(
                {task, relay_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket relay failed")
        finally:
            sink.close()
            await runner.close()
            for t in (task, relay_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(task, relay_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await ws.close()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
