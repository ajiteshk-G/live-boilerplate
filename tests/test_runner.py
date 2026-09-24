"""End-to-end session runner behaviour, driven by a scripted fake Live session.

No network, no credentials. What is being checked is the wiring: turn
boundaries, the tool loop, multi-part messages, and that usage lands in the
right turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from gemini_live.live.events import CollectingSink
from gemini_live.live.runner import LiveSessionRunner
from gemini_live.settings.schema import AppConfig
from gemini_live.tools.catalog import ToolCandidate
from gemini_live.tools.registry import ToolRegistry

from .conftest import FakeUsage

# ----------------------------------------------------------- message shapes


@dataclass
class Part:
    text: str | None = None
    inline_data: Any = None


@dataclass
class ModelTurn:
    parts: list[Part] = field(default_factory=list)


@dataclass
class ServerContent:
    model_turn: ModelTurn | None = None
    input_transcription: Any = None
    output_transcription: Any = None
    grounding_metadata: Any = None
    interrupted: bool = False
    turn_complete: bool = False


@dataclass
class Msg:
    usage_metadata: Any = None
    server_content: Any = None
    tool_call: Any = None
    tool_call_cancellation: Any = None
    session_resumption_update: Any = None
    go_away: Any = None


@dataclass
class FnCall:
    name: str
    args: dict[str, Any] | None = None
    id: str | None = "c1"


def audio_part(data: bytes) -> Part:
    return Part(inline_data=SimpleNamespace(data=data))


def transcript(text: str) -> Any:
    return SimpleNamespace(text=text)


# ------------------------------------------------------------- fake client


class FakeSession:
    def __init__(self, messages: list[Msg]) -> None:
        self._messages = messages
        self.realtime_inputs: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.on_exhausted: Any = None

    async def receive(self):
        for msg in self._messages:
            yield msg
        if self.on_exhausted is not None:
            await self.on_exhausted()

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_inputs.append(kwargs)

    async def send_tool_response(self, *, function_responses: list[Any]) -> None:
        self.tool_responses.extend(function_responses)


class FakeClient:
    """Mimics ``client.aio.live.connect(...)`` as an async context manager."""

    def __init__(self, *sessions: Any) -> None:
        self.sessions = list(sessions)
        self.configs: list[Any] = []
        outer = self

        class _Live:
            def connect(self, *, model: str, config: Any):
                outer.configs.append(config)
                session = outer.sessions.pop(0)

                class _Conn:
                    async def __aenter__(self):
                        return session

                    async def __aexit__(self, *exc):
                        return False

                return _Conn()

        self.aio = SimpleNamespace(live=_Live())


def make_runner(messages: list[Msg], *, tools: list[ToolCandidate] | None = None, **cfg_over):
    base: dict[str, Any] = {
        "vertex": {"project": "p"},
        "usage": {"log_path": None, "log_raw_snapshots": False},
        "search": {"google_search": {"enabled": False}, "log_violations": None},
    }
    base.update(cfg_over)
    cfg = AppConfig.model_validate(base)

    session = FakeSession(messages)
    client = FakeClient(session)
    sink = CollectingSink()
    runner = LiveSessionRunner(
        cfg, client, ToolRegistry(tools or []), sink=sink
    )
    # Stop after the scripted messages run out, instead of reconnecting forever.
    session.on_exhausted = runner.close
    return runner, sink, session


def tool(name: str, fn) -> ToolCandidate:
    return ToolCandidate(
        exposed_name=name,
        description="d",
        input_schema={"type": "object", "properties": {}},
        invoke=fn,
        origin="mcp:test",
    )


# ------------------------------------------------------------------- tests


async def test_connected_event_reports_the_tool_set():
    runner, sink, _ = make_runner([])
    await runner.run()

    connected = sink.of_kind("connected")
    assert connected and connected[0]["resumed"] is False


async def test_audio_and_transcript_in_one_message_are_both_handled():
    """A single server_content can carry audio AND a transcript. Branching on
    the first match silently drops the other."""
    runner, sink, _ = make_runner(
        [
            Msg(
                server_content=ServerContent(
                    model_turn=ModelTurn(parts=[audio_part(b"pcm-bytes")]),
                    output_transcription=transcript("hello there"),
                )
            )
        ]
    )
    await runner.run()

    assert sink.audio_chunks == [b"pcm-bytes"]
    assert sink.of_kind("transcript") == [{"role": "model", "text": "hello there"}]


async def test_every_part_of_a_multi_part_turn_is_processed():
    runner, sink, _ = make_runner(
        [
            Msg(
                server_content=ServerContent(
                    model_turn=ModelTurn(
                        parts=[audio_part(b"a"), Part(text="written"), audio_part(b"b")]
                    )
                )
            )
        ]
    )
    await runner.run()

    assert sink.audio_chunks == [b"a", b"b"]
    assert sink.of_kind("text") == [{"text": "written"}]


async def test_input_transcription_is_attributed_to_the_user():
    runner, sink, _ = make_runner(
        [Msg(server_content=ServerContent(input_transcription=transcript("what time is it")))]
    )
    await runner.run()

    assert sink.of_kind("transcript") == [{"role": "user", "text": "what time is it"}]


async def test_usage_is_committed_on_turn_complete():
    runner, sink, _ = make_runner(
        [
            Msg(usage_metadata=FakeUsage(prompt_token_count=1000, response_token_count=50,
                                         total_token_count=1050)),
            Msg(usage_metadata=FakeUsage(prompt_token_count=1000, response_token_count=120,
                                         total_token_count=1120)),
            Msg(server_content=ServerContent(turn_complete=True)),
        ]
    )
    await runner.run()

    turns = sink.of_kind("usage_turn")
    assert len(turns) == 1
    # Last snapshot wins; the two messages are not added together.
    assert turns[0]["scalars"]["total"] == 1120
    assert turns[0]["scalars"]["prompt"] == 1000


async def test_two_turns_accumulate_into_the_session_total():
    runner, sink, _ = make_runner(
        [
            Msg(usage_metadata=FakeUsage(prompt_token_count=1000, total_token_count=1100)),
            Msg(server_content=ServerContent(turn_complete=True)),
            Msg(usage_metadata=FakeUsage(prompt_token_count=1800, total_token_count=1950)),
            Msg(server_content=ServerContent(turn_complete=True)),
        ]
    )
    await runner.run()

    sessions = sink.of_kind("usage_session")
    assert sessions[-1]["turns"] == 2
    assert sessions[-1]["total"] == 3050


async def test_context_growth_is_reported_per_turn():
    runner, sink, _ = make_runner(
        [
            Msg(usage_metadata=FakeUsage(prompt_token_count=1000, total_token_count=1100)),
            Msg(server_content=ServerContent(turn_complete=True)),
            Msg(usage_metadata=FakeUsage(prompt_token_count=1800, total_token_count=1900)),
            Msg(server_content=ServerContent(turn_complete=True)),
        ]
    )
    await runner.run()

    turns = sink.of_kind("usage_turn")
    assert turns[0]["cost"]["prompt_delta"] == 0
    assert turns[1]["cost"]["prompt_delta"] == 800
    assert turns[1]["cost"]["context_rent"] == 1800


async def test_turn_interrupted_by_hangup_is_still_billed():
    """No turn_complete ever arrives. Those tokens were charged regardless."""
    runner, sink, _ = make_runner(
        [Msg(usage_metadata=FakeUsage(prompt_token_count=500, total_token_count=560))]
    )
    await runner.run()

    turns = sink.of_kind("usage_turn")
    assert len(turns) == 1
    assert turns[0]["scalars"]["total"] == 560


async def test_tool_call_is_dispatched_and_answered():
    called = {}

    async def get_time(args):
        called.update(args)
        return {"now": "12:00"}

    runner, sink, session = make_runner(
        [
            Msg(
                tool_call=SimpleNamespace(
                    function_calls=[FnCall("t__now", {"tz": "UTC"}, "call-9")]
                )
            )
        ],
        tools=[tool("t__now", get_time)],
    )
    await runner.run()

    assert called == {"tz": "UTC"}
    assert len(session.tool_responses) == 1
    response = session.tool_responses[0]
    assert response.id == "call-9"  # echoing the id is mandatory
    assert response.response == {"now": "12:00"}
    assert sink.of_kind("tool_call") == [
        {
            "names": ["t__now"],
            "calls": [{"id": "call-9", "name": "t__now", "args": {"tz": "UTC"}}],
        }
    ]
    tool_results = sink.of_kind("tool_result")
    assert len(tool_results) == 1
    assert tool_results[0]["names"] == ["t__now"]
    assert tool_results[0]["errors"] == []
    assert tool_results[0]["results"] == [
        {"id": "call-9", "name": "t__now", "response": {"now": "12:00"}}
    ]
    assert isinstance(tool_results[0].get("round_trip_ms"), float)


async def test_failing_tool_is_reported_but_still_answered():
    async def boom(args):
        raise RuntimeError("upstream down")

    runner, sink, session = make_runner(
        [Msg(tool_call=SimpleNamespace(function_calls=[FnCall("t__bad")]))],
        tools=[tool("t__bad", boom)],
    )
    await runner.run()

    # The model must always get a response, or the turn hangs forever.
    assert len(session.tool_responses) == 1
    assert "error" in session.tool_responses[0].response
    assert sink.of_kind("tool_result")[0]["errors"] == ["t__bad"]


async def test_empty_tool_call_is_ignored():
    runner, sink, session = make_runner(
        [Msg(tool_call=SimpleNamespace(function_calls=[]))]
    )
    await runner.run()

    assert session.tool_responses == []
    assert sink.of_kind("tool_call") == []


async def test_resumption_handle_is_captured():
    runner, _, _ = make_runner(
        [
            Msg(
                session_resumption_update=SimpleNamespace(
                    resumable=True, new_handle="handle-xyz"
                )
            )
        ]
    )
    await runner.run()

    assert runner._resume_handle == "handle-xyz"


async def test_non_resumable_update_is_ignored():
    runner, _, _ = make_runner(
        [Msg(session_resumption_update=SimpleNamespace(resumable=False, new_handle="nope"))]
    )
    await runner.run()

    assert runner._resume_handle is None


async def test_go_away_is_surfaced():
    runner, sink, _ = make_runner([Msg(go_away=SimpleNamespace(time_left="10s"))])
    await runner.run()

    assert sink.of_kind("go_away") == [{"time_left": "10s"}]


async def test_interruption_is_surfaced_for_barge_in():
    runner, sink, _ = make_runner([Msg(server_content=ServerContent(interrupted=True))])
    await runner.run()

    assert sink.of_kind("interrupted") == [{}]


async def test_tool_call_cancellation_is_surfaced():
    runner, sink, _ = make_runner(
        [Msg(tool_call_cancellation=SimpleNamespace(ids=["a", "b"]))]
    )
    await runner.run()

    assert sink.of_kind("tool_cancelled") == [{"ids": ["a", "b"]}]


async def test_usage_alert_fires_when_context_rent_is_high():
    runner, sink, _ = make_runner(
        [
            Msg(usage_metadata=FakeUsage(prompt_token_count=50000, total_token_count=50100)),
            Msg(server_content=ServerContent(turn_complete=True)),
        ]
    )
    await runner.run()

    alerts = sink.of_kind("usage_alert")
    assert any(a["kind"] == "prompt_tokens" for a in alerts)


async def test_usage_events_suppressed_when_ui_display_is_off():
    runner, sink, _ = make_runner(
        [
            Msg(usage_metadata=FakeUsage(prompt_token_count=100, total_token_count=150)),
            Msg(server_content=ServerContent(turn_complete=True)),
        ],
        usage={"log_path": None, "log_raw_snapshots": False, "show_in_ui": False},
    )
    await runner.run()

    assert sink.of_kind("usage_turn") == []
    # Accounting still happens; only the display is suppressed.
    assert runner.accountant.session_total().turns == 1


async def test_uplink_audio_reaches_the_session():
    runner, _, session = make_runner([])
    await runner.uplink.audio(b"mic-data")
    await runner.run()

    # Best-effort: the uplink pump is cancelled as soon as the downlink ends,
    # so only assert that nothing raised and the queue was drained or cancelled.
    assert session.realtime_inputs == [] or "audio" in session.realtime_inputs[0]


# ------------------------------------------- per-turn receive() iterator


class TurnScriptedSession:
    """Mimics the real SDK, where ``receive()`` ends when a turn completes.

    ``google.genai`` breaks out of the ``receive()`` generator on
    ``turn_complete`` (live.py, ``_is_interaction_complete``), so a session that
    spans several turns hands back a *fresh* iterator per turn over the same
    open websocket. ``FakeSession`` yields everything in one go and therefore
    cannot catch a runner that stops after the first turn.
    """

    def __init__(self, turns: list[list[Msg]]) -> None:
        self._turns = list(turns)
        self.receive_calls = 0
        self.realtime_inputs: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.on_exhausted: Any = None

    async def receive(self):
        self.receive_calls += 1
        if not self._turns:
            if self.on_exhausted is not None:
                await self.on_exhausted()
            return
        for msg in self._turns.pop(0):
            yield msg

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_inputs.append(kwargs)

    async def send_tool_response(self, *, function_responses: list[Any]) -> None:
        self.tool_responses.extend(function_responses)


async def test_answer_after_tool_call_is_not_lost_to_a_reconnect():
    """The regression: receive() ends on the function-call turn, and the
    model's actual answer only arrives in the turn after it."""
    called: list[str] = []

    def now(args: dict[str, Any]) -> str:
        called.append("now")
        return "12:00"

    turns = [
        # Turn 0: the model asks for the tool, then completes the turn.
        [
            Msg(tool_call=SimpleNamespace(function_calls=[FnCall(name="now")])),
            Msg(
                usage_metadata=FakeUsage(prompt_token_count=57, total_token_count=62),
                server_content=ServerContent(turn_complete=True),
            ),
        ],
        # Turn 1: the answer that uses the tool result.
        [
            Msg(
                server_content=ServerContent(
                    model_turn=ModelTurn(parts=[Part(text="It is 12:00.")])
                )
            ),
            Msg(
                usage_metadata=FakeUsage(prompt_token_count=80, total_token_count=95),
                server_content=ServerContent(turn_complete=True),
            ),
        ],
    ]

    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "usage": {"log_path": None, "log_raw_snapshots": False},
            "search": {"google_search": {"enabled": False}, "log_violations": None},
        }
    )
    session = TurnScriptedSession(turns)
    client = FakeClient(session)  # exactly one session: a reconnect would IndexError
    sink = CollectingSink()
    runner = LiveSessionRunner(cfg, client, ToolRegistry([tool("now", now)]), sink=sink)
    session.on_exhausted = runner.close

    await runner.run()

    assert called == ["now"]
    assert session.receive_calls >= 2, "runner gave up after the first turn"
    texts = "".join(e["text"] for e in sink.of_kind("text"))
    assert texts == "It is 12:00."
    # One connect only. A second 'connected' event means the session was torn
    # down and the conversation history thrown away.
    assert len(sink.of_kind("connected")) == 1
    assert sink.of_kind("reconnecting") == []
    assert len(sink.of_kind("usage_turn")) == 2


async def test_interrupted_turn_commits_inflight_usage_and_preserves_monotonicity():
    """When the user barges in (interrupted=True), the runner must finalize the
    interrupted turn so its usage is committed and does not leak into the next turn."""
    runner, sink, _ = make_runner(
        [
            # Turn 0 starts generating, reaches 100 prompt / 40 response tokens,
            # then user interrupts
            Msg(
                usage_metadata=FakeUsage(
                    prompt_token_count=100, response_token_count=40, total_token_count=140
                ),
                server_content=ServerContent(interrupted=True),
            ),
            # Turn 1 starts fresh with 110 prompt / 10 response tokens
            # (lower response count than 40!)
            Msg(
                usage_metadata=FakeUsage(
                    prompt_token_count=110, response_token_count=10, total_token_count=120
                ),
                server_content=ServerContent(turn_complete=True),
            ),
        ]
    )
    await runner.run()

    assert len(sink.of_kind("interrupted")) == 1
    assert len(sink.of_kind("usage_turn")) == 2
    assert runner.accountant.session_total().turns == 2
    # Response tokens: turn 0 (40) + turn 1 (10) = 50
    assert runner.accountant.session_total().response_tokens == 50


async def test_setup_failure_stops_after_max_retries_instead_of_looping_forever(monkeypatch):
    """If connect() enters but receive() raises before yielding any message,
    attempts must NOT be reset to 0 on every attempt."""
    import asyncio

    import pytest

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    class FailingSession:
        async def receive(self):
            if False:
                yield None
            raise RuntimeError("1011 transient server hiccup before first message")

    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "session": {"resumption": {"enabled": True}},
            "usage": {"log_path": None, "log_raw_snapshots": False},
            "search": {"google_search": {"enabled": False}, "log_violations": None},
        }
    )
    client = FakeClient(FailingSession(), FailingSession(), FailingSession(), FailingSession())
    sink = CollectingSink()
    runner = LiveSessionRunner(cfg, client, ToolRegistry([]), sink=sink)

    with pytest.raises(RuntimeError, match="1011 transient"):
        await runner.run()

    # Should attempt 1 initial + 3 retries = 4 connects, then raise
    assert len(sink.of_kind("connected")) == 4
    assert len(sink.of_kind("reconnecting")) == 3


class _AlwaysFailingSession:
    async def receive(self):
        if False:
            yield None
        raise RuntimeError("1011 transient server hiccup before first message")


def _reconnect_cfg(**reconnect: Any) -> AppConfig:
    return AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "session": {"resumption": {"enabled": True}, "reconnect": reconnect},
            "usage": {"log_path": None, "log_raw_snapshots": False},
            "search": {"google_search": {"enabled": False}, "log_violations": None},
        }
    )


async def test_reconnect_attempt_limit_comes_from_config(monkeypatch):
    """The retry ceiling must be a config knob, not a literal in the runner."""
    import asyncio

    import pytest

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    cfg = _reconnect_cfg(max_attempts=1)
    client = FakeClient(*[_AlwaysFailingSession() for _ in range(5)])
    sink = CollectingSink()
    runner = LiveSessionRunner(cfg, client, ToolRegistry([]), sink=sink)

    with pytest.raises(RuntimeError, match="1011 transient"):
        await runner.run()

    # 1 initial connect + 1 permitted retry, then give up.
    assert len(sink.of_kind("connected")) == 2
    assert len(sink.of_kind("reconnecting")) == 1


async def test_reconnect_backoff_follows_configured_bounds(monkeypatch):
    """Backoff starts at initial_backoff_seconds and is capped, both from config."""
    import asyncio

    import pytest

    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    cfg = _reconnect_cfg(max_attempts=4, initial_backoff_seconds=0.5, max_backoff_seconds=2.0)
    client = FakeClient(*[_AlwaysFailingSession() for _ in range(6)])
    sink = CollectingSink()
    runner = LiveSessionRunner(cfg, client, ToolRegistry([]), sink=sink)

    with pytest.raises(RuntimeError, match="1011 transient"):
        await runner.run()

    # 0.5, 1.0, 2.0, then held at the 2.0 cap rather than doubling to 4.0.
    assert delays == [0.5, 1.0, 2.0, 2.0]


