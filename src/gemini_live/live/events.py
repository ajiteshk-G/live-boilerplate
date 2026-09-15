"""Events emitted by the session runner toward a client (browser or CLI)."""

from __future__ import annotations

from typing import Any, Protocol


class EventSink(Protocol):
    """Where a running session sends audio and structured events."""

    async def audio(self, pcm: bytes) -> None:
        """Model audio: PCM16 mono @ 24 kHz."""

    async def event(self, kind: str, payload: dict[str, Any]) -> None:
        """A structured JSON event."""


class NullSink:
    async def audio(self, pcm: bytes) -> None:  # noqa: D102
        return None

    async def event(self, kind: str, payload: dict[str, Any]) -> None:  # noqa: D102
        return None


class CollectingSink:
    """Test/CLI sink that records everything it receives."""

    def __init__(self) -> None:
        self.audio_chunks: list[bytes] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def audio(self, pcm: bytes) -> None:
        self.audio_chunks.append(pcm)

    async def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, payload))

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [payload for k, payload in self.events if k == kind]
