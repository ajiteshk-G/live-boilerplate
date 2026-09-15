"""WebSocket wire protocol between the browser and the relay.

Binary frames carry audio (no base64 tax); JSON frames carry control and
telemetry. Simple enough to debug directly in devtools.

    -> server  binary   mic PCM16 mono @ 16 kHz
    -> server  {"type":"text","text":...}
    -> server  {"type":"mic","on":false}        triggers audio_stream_end
    <- client  binary   model PCM16 mono @ 24 kHz
    <- client  {"type":"transcript"|"usage_turn"|"citations"|"interrupted"|...}
"""

from __future__ import annotations

import json
from typing import Any


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
