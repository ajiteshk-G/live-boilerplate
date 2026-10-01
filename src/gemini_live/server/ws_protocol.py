"""WebSocket wire protocol between the browser and the relay.

Binary frames carry audio (no base64 tax); JSON frames carry control and
telemetry. Simple enough to debug directly in devtools.

    -> server  binary   mic PCM16 mono @ 16 kHz
    -> server  {"type":"text","text":...}
    -> server  {"type":"mic","on":false}        triggers audio_stream_end
    -> server  {"type":"end_call"}              hangs up: closes the Live session
    <- client  binary   model PCM16 mono @ 24 kHz
    <- client  {"type":"transcript"|"usage_turn"|"citations"|"interrupted"|...}
    <- client  {"type":"session_ended","reason":...}
"""

from __future__ import annotations

import json
from typing import Any, Protocol


def encode_event(kind: str, payload: dict[str, Any]) -> str:
    return json.dumps({"type": kind, **payload})


def decode_client_message(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except ValueError:
        return {"type": "unknown"}
    if not isinstance(data, dict):
        return {"type": "unknown"}
    data.setdefault("type", "unknown")
    return data


class _RunnerLike(Protocol):
    """The slice of ``LiveSessionRunner`` this module needs."""

    uplink: Any

    async def close(self) -> None: ...


async def apply_client_message(payload: dict[str, Any], runner: _RunnerLike) -> bool:
    """Apply one decoded client message.

    Returns False when the client hung up, so the caller stops relaying. Kept
    here rather than inline in the endpoint so the control protocol is testable
    without a live socket.
    """
    kind = payload.get("type")
    if kind == "text" and payload.get("text"):
        await runner.uplink.text(str(payload["text"]))
    elif kind == "client_content" and payload.get("text"):
        await runner.uplink.client_content(
            str(payload["text"]),
            role=str(payload.get("role") or "user"),
            turn_complete=bool(payload.get("turn_complete", True)),
        )
    elif kind == "mic" and payload.get("on") is False:
        await runner.uplink.mic_off()
    elif kind == "activity_start":
        await runner.uplink.activity_start()
    elif kind == "activity_end":
        await runner.uplink.activity_end()
    elif kind == "end_call":
        # Stop means stop: end the Live session rather than just muting, so the
        # model stops generating and the session stops being billed.
        await runner.close()
        return False
    return True

