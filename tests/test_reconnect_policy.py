"""Reconnect policy: which failures are worth retrying.

Retrying a permanent error wastes the user's time and buries the one message
that tells them what to fix. Retrying too little breaks a recoverable call.
"""

from __future__ import annotations

import pytest

from gemini_live.live.runner import is_permanent_error


@pytest.mark.parametrize(
    "message",
    [
        # The exact failure seen when the API is not enabled on a project.
        "1008 None. Agent Platform API has not been used in project demo before or it is disabled.",
        "403 PERMISSION_DENIED",
        "401 UNAUTHENTICATED: credentials missing",
        "400 INVALID_ARGUMENT: bad model config",
        "404 Publisher Model was not found",
        "FAILED_PRECONDITION: billing is not enabled",
        "Quota exceeded for this project",
        # Observed for real: a native-audio model rejecting TEXT output. The
        # config is wrong for that model and no amount of retrying fixes it.
        "1007 None. Text output is not supported for native audio output model.",
    ],
)
def test_permanent_failures_are_recognised(message):
    assert is_permanent_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "connection reset by peer",
        "1011 internal error",
        "timed out waiting for the server",
        "503 Service Unavailable",
        "sent 1000 (OK); no close frame received",
        "",
    ],
)
def test_transient_failures_are_still_retried(message):
    """Default must be transient: wrongly giving up on a blip is the worse
    failure mode."""
    assert is_permanent_error(RuntimeError(message)) is False


def test_matching_is_case_insensitive():
    assert is_permanent_error(RuntimeError("Permission_Denied")) is True


def test_exception_group_is_unwrapped():
    """TaskGroup wraps downlink errors in ExceptionGroup; permanent sub-exceptions
    must be caught."""
    group = ExceptionGroup(
        "unhandled errors in a taskgroup (1 sub-exception)",
        [RuntimeError("400 Invalid argument: bad setup")],
    )
    assert is_permanent_error(group) is True

    transient_group = ExceptionGroup(
        "unhandled errors in a taskgroup (1 sub-exception)",
        [RuntimeError("1011 internal error")],
    )
    assert is_permanent_error(transient_group) is False


async def test_uplink_drain_audio_removes_only_audio():
    import asyncio

    from gemini_live.live.runner import _Uplink

    up = _Uplink(asyncio.Queue())
    await up.audio(b"chunk1")
    await up.text("hello")
    await up.audio(b"chunk2")
    await up.mic_off()

    dropped = up.drain_audio()
    assert dropped == 2
    remaining = []
    while not up.queue.empty():
        remaining.append(up.queue.get_nowait()["kind"])
    assert remaining == ["text", "audio_stream_end"]

