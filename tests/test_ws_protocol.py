"""Client -> server control protocol, in particular hanging up.

The bug this pins down: "Stop" used to only mute the microphone, leaving the
Live session open and billing. Stop must end the session.
"""

from __future__ import annotations

from typing import Any

import pytest

from gemini_live.server.ws_protocol import (
    apply_client_message,
    decode_client_message,
    encode_event,
)


class FakeUplink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.mic_offs = 0
        self.audio: list[bytes] = []

    async def text(self, text: str) -> None:
        self.texts.append(text)

    async def mic_off(self) -> None:
        self.mic_offs += 1


class FakeRunner:
    def __init__(self) -> None:
        self.uplink = FakeUplink()
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_end_call_closes_the_runner_and_stops_the_relay():
    runner = FakeRunner()

    keep_going = await apply_client_message({"type": "end_call"}, runner)

    assert keep_going is False
    assert runner.closed == 1


@pytest.mark.asyncio
async def test_mic_off_only_ends_the_utterance_and_keeps_the_call_up():
    """Muting is not hanging up: the session must survive it."""
    runner = FakeRunner()

    keep_going = await apply_client_message({"type": "mic", "on": False}, runner)

    assert keep_going is True
    assert runner.uplink.mic_offs == 1
    assert runner.closed == 0


@pytest.mark.asyncio
async def test_text_is_forwarded_and_the_call_continues():
    runner = FakeRunner()

    keep_going = await apply_client_message({"type": "text", "text": "hello"}, runner)

    assert keep_going is True
    assert runner.uplink.texts == ["hello"]
    assert runner.closed == 0


@pytest.mark.asyncio
async def test_unknown_and_empty_messages_are_ignored_without_ending_the_call():
    runner = FakeRunner()

    for payload in ({"type": "nonsense"}, {"type": "text", "text": ""}, {}):
        assert await apply_client_message(payload, runner) is True
    assert runner.closed == 0
    assert runner.uplink.texts == []


def test_malformed_json_does_not_raise():
    assert decode_client_message("not json")["type"] == "unknown"
    assert decode_client_message("[1,2,3]")["type"] == "unknown"


def test_session_ended_is_encodable_for_the_browser():
    event: Any = encode_event("session_ended", {"reason": "ended by the user"})
    assert '"type": "session_ended"' in event
    assert "ended by the user" in event
