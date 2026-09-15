"""Tool dispatch.

The Live API never executes tools for you: it emits a toolCall and waits. The
contract that matters most is echoing the call id -- omitting it wedges the turn,
and it is the single most common Live API bug.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from gemini_live.tools.catalog import ToolCandidate
from gemini_live.tools.registry import ToolRegistry


@dataclass
class FakeCall:
    """Stand-in for ``types.FunctionCall``."""

    name: str
    args: dict[str, Any] | None = None
    id: str | None = "call-1"


def tool(name: str, fn) -> ToolCandidate:
    return ToolCandidate(
        exposed_name=name,
        description="d",
        input_schema={"type": "object", "properties": {}},
        invoke=fn,
        origin="mcp:test",
    )


async def test_call_id_is_echoed():
    """Mandatory: the API matches responses to calls by id."""

    async def ok(args):
        return {"result": 42}

    registry = ToolRegistry([tool("t", ok)])
    response = await registry.dispatch(FakeCall(name="t", id="abc-123"))

    assert response.id == "abc-123"
    assert response.name == "t"
    assert response.response == {"result": 42}


async def test_missing_call_id_is_passed_through_as_none():
    """Some backends omit the id. We must not invent one."""

    async def ok(args):
        return {}

    registry = ToolRegistry([tool("t", ok)])
    assert (await registry.dispatch(FakeCall(name="t", id=None))).id is None


async def test_arguments_are_forwarded():
    seen = {}

    async def capture(args):
        seen.update(args)
        return {"ok": True}

    registry = ToolRegistry([tool("t", capture)])
    await registry.dispatch(FakeCall(name="t", args={"q": "hello", "n": 3}))

    assert seen == {"q": "hello", "n": 3}


async def test_absent_arguments_become_an_empty_dict():
    async def needs_dict(args):
        assert args == {}
        return {"ok": True}

    registry = ToolRegistry([tool("t", needs_dict)])
    assert (await registry.dispatch(FakeCall(name="t", args=None))).response == {"ok": True}


# ---------------------------------------------------------------- failures


async def test_unknown_tool_returns_an_error_rather_than_raising():
    """Models do invent tool names, and a curated session deliberately omits
    most of the catalog. Crashing would end the call."""
    registry = ToolRegistry([])
    response = await registry.dispatch(FakeCall(name="ghost"))

    assert "error" in response.response
    assert "not available" in response.response["error"]
    assert response.id == "call-1"  # still answered, so the turn can complete


async def test_tool_exception_is_reported_back_to_the_model():
    async def boom(args):
        raise ValueError("database offline")

    registry = ToolRegistry([tool("t", boom)])
    response = await registry.dispatch(FakeCall(name="t"))

    assert response.response["error"] == "ValueError: database offline"


async def test_tool_timeout_is_reported_and_does_not_hang_the_session():
    async def slow(args):
        await asyncio.sleep(10)
        return {}

    registry = ToolRegistry([tool("t", slow)], timeout_s=0.05)
    response = await registry.dispatch(FakeCall(name="t"))

    assert "timed out" in response.response["error"]


async def test_dispatch_never_raises_whatever_the_tool_does():
    async def nasty(args):
        raise BaseException("not even an Exception")  # noqa: TRY002

    registry = ToolRegistry([tool("t", nasty)])
    with pytest.raises(BaseException, match="not even"):
        # BaseException is intentionally NOT swallowed -- that would hide
        # KeyboardInterrupt and CancelledError.
        await registry.dispatch(FakeCall(name="t"))


# ------------------------------------------------------------- parallel


async def test_multiple_calls_are_dispatched_concurrently():
    """The model can request several tools in one turn; running them serially
    would multiply the user-perceived latency."""
    order = []

    async def slow(args):
        await asyncio.sleep(0.05)
        order.append("slow")
        return {"t": "slow"}

    async def fast(args):
        order.append("fast")
        return {"t": "fast"}

    registry = ToolRegistry([tool("slow", slow), tool("fast", fast)])
    responses = await registry.dispatch_all(
        [FakeCall(name="slow", id="1"), FakeCall(name="fast", id="2")]
    )

    assert order == ["fast", "slow"]  # ran in parallel, not in call order
    # Responses stay in the order the model asked for.
    assert [r.id for r in responses] == ["1", "2"]


async def test_one_failing_tool_does_not_break_the_others():
    async def ok(args):
        return {"ok": True}

    async def boom(args):
        raise RuntimeError("nope")

    registry = ToolRegistry([tool("ok", ok), tool("bad", boom)])
    responses = await registry.dispatch_all(
        [FakeCall(name="ok", id="1"), FakeCall(name="bad", id="2")]
    )

    assert responses[0].response == {"ok": True}
    assert "error" in responses[1].response


async def test_dispatch_all_on_empty_list():
    assert await ToolRegistry([]).dispatch_all([]) == []


# --------------------------------------------------------- declarations


def test_declarations_bundle_into_a_single_tool():
    async def noop(args):
        return {}

    registry = ToolRegistry([tool("a", noop), tool("b", noop)])
    decls = registry.declarations()

    assert len(decls) == 1  # one types.Tool holding both declarations
    assert len(decls[0].function_declarations) == 2
    assert {d.name for d in decls[0].function_declarations} == {"a", "b"}


def test_empty_registry_declares_nothing():
    """Must be [] and not a Tool with an empty list, which the API rejects."""
    assert ToolRegistry([]).declarations() == []


def test_registry_reports_its_names():
    async def noop(args):
        return {}

    registry = ToolRegistry([tool("z", noop), tool("a", noop)])
    assert registry.names == ["a", "z"]
    assert len(registry) == 2
