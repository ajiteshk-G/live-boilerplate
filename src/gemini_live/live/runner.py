"""The Live session runner: uplink/downlink pumps, tool loop, token accounting."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from ..search.domain_policy import DomainPolicy
from ..search.grounding_filter import GroundingFilter
from ..settings.schema import AppConfig
from ..tools.registry import ToolRegistry
from ..usage.accountant import TokenAccountant
from ..usage.cost_model import CostModel
from ..usage.pricing import PriceTable
from .connect_config import build_live_config
from .events import EventSink, NullSink

log = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
INPUT_MIME = f"audio/pcm;rate={INPUT_SAMPLE_RATE}"

# Substrings that mean "this will fail identically on every retry". Matching is
# deliberately conservative: anything not listed is treated as transient and
# retried, because wrongly giving up on a recoverable blip is worse than a few
# wasted seconds.
_PERMANENT_MARKERS = (
    "permission_denied",
    "unauthenticated",
    "invalid authentication credentials",
    "invalid_argument",
    "invalid argument",
    "failed_precondition",
    "has not been used in project",
    "api has not been used",
    "is disabled",
    "was not found",
    "not found for api version",
    # Capability mismatches: e.g. "Text output is not supported for native audio
    # output model". The config is wrong for this model and always will be.
    "is not supported",
    "not supported for",
    "billing",
    "quota exceeded",
    "403",
    "401",
    "404",
)

PERMANENT_HINT = (
    "This will not succeed on retry. Check, in order: "
    "(1) the Vertex AI API is enabled for your project, "
    "(2) `gcloud auth application-default login` is current, "
    "(3) vertex.project and vertex.location are correct, "
    "(4) model.name exists in that location. "
    "Run `glive doctor` for a guided check."
)


def is_permanent_error(exc: BaseException) -> bool:
    """Whether reconnecting could plausibly help.

    Config and credential failures surface as a WebSocket close or inside an
    ExceptionGroup from TaskGroup, so we unwrap groups and causes recursively.
    """
    if isinstance(exc, BaseExceptionGroup):
        return any(is_permanent_error(sub) for sub in exc.exceptions)
    text = str(exc).lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return True
    if exc.__cause__ is not None and is_permanent_error(exc.__cause__):
        return True
    return exc.__context__ is not None and is_permanent_error(exc.__context__)


@dataclass
class _Uplink:
    """Queue of client -> model messages."""

    queue: asyncio.Queue[dict[str, Any]]

    async def audio(self, pcm: bytes) -> None:
        await self.queue.put({"kind": "audio", "data": pcm})

    async def text(self, text: str) -> None:
        await self.queue.put({"kind": "text", "text": text})

    async def mic_off(self) -> None:
        await self.queue.put({"kind": "audio_stream_end"})

    async def close(self) -> None:
        await self.queue.put({"kind": "close"})

    def drain_audio(self) -> int:
        """Remove queued audio frames so stale mic chunks don't flood a resumed session."""
        kept: list[dict[str, Any]] = []
        dropped = 0
        while not self.queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                item = self.queue.get_nowait()
                if item.get("kind") == "audio":
                    dropped += 1
                else:
                    kept.append(item)
        for item in kept:
            self.queue.put_nowait(item)
        return dropped


class LiveSessionRunner:
    """Owns one logical conversation, across reconnects."""

    def __init__(
        self,
        cfg: AppConfig,
        client: Any,
        registry: ToolRegistry,
        *,
        sink: EventSink | None = None,
        accountant: TokenAccountant | None = None,
        cost_model: CostModel | None = None,
        policy: DomainPolicy | None = None,
        grounding: GroundingFilter | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._registry = registry
        self._sink: EventSink = sink or NullSink()
        self._policy = policy or DomainPolicy(cfg.search.domains.allow, cfg.search.domains.deny)
        self._grounding = grounding or GroundingFilter(
            self._policy, violations_log=cfg.search.log_violations
        )
        self._accountant = accountant or TokenAccountant(
            mode=cfg.usage.accounting_mode,
            raw_log_path=cfg.usage.log_path,
            log_raw_snapshots=cfg.usage.log_raw_snapshots,
        )
        prices = (
            PriceTable(
                text_input_per_1k=cfg.usage.pricing.text_input_per_1k,
                audio_input_per_1k=cfg.usage.pricing.audio_input_per_1k,
                text_output_per_1k=cfg.usage.pricing.text_output_per_1k,
                audio_output_per_1k=cfg.usage.pricing.audio_output_per_1k,
                thinking_per_1k=cfg.usage.pricing.thinking_per_1k,
                currency=cfg.usage.pricing.currency,
            )
            if cfg.usage.pricing is not None
            else None
        )
        self._cost = cost_model or CostModel(
            tool_declaration_tokens=registry.total_declaration_tokens,
            warn_prompt_tokens_per_turn=cfg.usage.alerts.warn_prompt_tokens_per_turn,
            warn_rent_ratio=cfg.usage.alerts.warn_rent_ratio,
            warn_session_total=cfg.usage.alerts.warn_session_total,
            prices=prices,
        )

        self.uplink = _Uplink(asyncio.Queue())
        self._session: Any | None = None
        self._resume_handle: str | None = None
        self._closed = False
        self._received_any = False
        self._pending_tools: set[str] = set()
        self._tool_tasks: dict[str, asyncio.Task[Any]] = {}

    # ------------------------------------------------------------------- run

    async def run(self) -> None:
        """Connect and pump until closed. Reconnects transparently when the
        server sends GoAway and session resumption is enabled."""
        attempts = 0
        try:
            while not self._closed:
                self._received_any = False
                config = build_live_config(
                    self._cfg,
                    self._registry,
                    policy=self._policy,
                    resumption_handle=self._resume_handle,
                )
                try:
                    async with self._client.aio.live.connect(
                        model=self._cfg.model.name, config=config
                    ) as session:
                        self._session = session
                        await self._sink.event(
                            "connected",
                            {
                                "model": self._cfg.model.name,
                                "tools": self._registry.names,
                                "resumed": self._resume_handle is not None,
                                "language": self._cfg.speech.language_code,
                                "language_mode": self._cfg.speech.language_mode,
                            },
                        )
                        await self._pump(session)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._finalize_turn()
                    if self._closed:
                        break
                    if is_permanent_error(exc):
                        # Retrying cannot help, and backing off three times just
                        # buries the one message that tells the user what to fix.
                        log.error("live session failed permanently: %s", exc)
                        await self._sink.event(
                            "error",
                            {
                                "message": f"{type(exc).__name__}: {exc}",
                                "permanent": True,
                                "hint": PERMANENT_HINT,
                            },
                        )
                        raise
                    if self._received_any:
                        attempts = 0
                    attempts += 1
                    rc = self._cfg.session.reconnect
                    if not self._cfg.session.resumption.enabled or attempts > rc.max_attempts:
                        await self._sink.event(
                            "error", {"message": f"{type(exc).__name__}: {exc}"}
                        )
                        raise
                    self.uplink.drain_audio()
                    delay = min(
                        rc.initial_backoff_seconds * 2 ** (attempts - 1),
                        rc.max_backoff_seconds,
                    )
                    log.warning("live session dropped (%s); reconnecting in %ss", exc, delay)
                    await self._sink.event(
                        "reconnecting", {"in_seconds": delay, "reason": str(exc)}
                    )
                    await asyncio.sleep(delay)
                finally:
                    self._session = None
        finally:
            # A call that ends mid-generation should still be billed.
            await self._finalize_turn()

    async def _pump(self, session: Any) -> None:
        up = asyncio.create_task(self._pump_uplink(session))
        down = asyncio.create_task(self._pump_downlink(session))
        try:
            done, pending = await asyncio.wait(
                {up, down}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._closed:
                await self._wait_tool_tasks()
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
        finally:
            for task in (up, down):
                if not task.done():
                    task.cancel()
            await asyncio.gather(up, down, return_exceptions=True)

    # --------------------------------------------------------------- uplink

    async def _pump_uplink(self, session: Any) -> None:
        from google.genai import types

        while True:
            msg = await self.uplink.queue.get()
            kind = msg.get("kind")
            if kind == "close":
                self._closed = True
                return
            if kind == "audio":
                await session.send_realtime_input(
                    audio=types.Blob(data=msg["data"], mime_type=INPUT_MIME)
                )
            elif kind == "text":
                # send_realtime_input for live input; send_client_content is only
                # for seeding initial history.
                await session.send_realtime_input(text=msg["text"])
            elif kind == "audio_stream_end":
                await session.send_realtime_input(audio_stream_end=True)

    # -------------------------------------------------------------- downlink

    async def _pump_downlink(self, session: Any) -> None:
        # `session.receive()` is a PER-TURN iterator, not a session-lifetime one:
        # the SDK breaks out of the generator as soon as a turn completes (see
        # google/genai/live.py, `_is_interaction_complete`). Letting this method
        # return at that point would look like a dropped connection to `run()`,
        # which would reconnect and discard the whole conversation after every
        # single turn -- and would swallow the model's answer to a tool call,
        # since that answer arrives in the turn AFTER the function call.
        # So we re-enter the iterator for each turn and let a genuinely closed
        # websocket surface as an exception instead.
        try:
            while not self._closed:
                turn_messages = 0
                async for message in session.receive():
                    self._received_any = True
                    turn_messages += 1
                    await self._handle_message(message)
                await self._wait_tool_tasks()
                if turn_messages == 0:
                    # The generator ended without yielding anything, which means the
                    # stream is finished rather than the turn. Stop, don't hot-loop.
                    return
        finally:
            self._cancel_tool_tasks()

    async def _handle_message(self, message: Any) -> None:
        # usage_metadata can ride on ANY server message.
        if getattr(message, "usage_metadata", None) is not None:
            self._accountant.observe(message.usage_metadata)

        content = getattr(message, "server_content", None)
        if content is not None:
            await self._handle_server_content(content)

        tool_call = getattr(message, "tool_call", None)
        if tool_call is not None:
            await self._handle_tool_call(tool_call)

        cancellation = getattr(message, "tool_call_cancellation", None)
        if cancellation is not None:
            ids = [str(i) for i in (getattr(cancellation, "ids", None) or [])]
            self._pending_tools.difference_update(ids)
            for cid in ids:
                t = self._tool_tasks.pop(cid, None)
                if t is not None and not t.done():
                    t.cancel()
            await self._sink.event("tool_cancelled", {"ids": ids})

        update = getattr(message, "session_resumption_update", None)
        if update is not None and getattr(update, "resumable", False):
            self._resume_handle = getattr(update, "new_handle", None)

        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            await self._sink.event(
                "go_away", {"time_left": str(getattr(go_away, "time_left", ""))}
            )

    async def _handle_server_content(self, content: Any) -> None:
        # A single event can carry multiple parts (audio AND transcript), so we
        # must process all of them rather than branching on the first match.
        model_turn = getattr(content, "model_turn", None)
        if model_turn is not None:
            for part in getattr(model_turn, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    await self._sink.audio(inline.data)
                text = getattr(part, "text", None)
                if text:
                    await self._sink.event("text", {"text": text})

        input_tx = getattr(content, "input_transcription", None)
        if input_tx is not None and getattr(input_tx, "text", None):
            await self._sink.event("transcript", {"role": "user", "text": input_tx.text})

        output_tx = getattr(content, "output_transcription", None)
        if output_tx is not None and getattr(output_tx, "text", None):
            await self._sink.event("transcript", {"role": "model", "text": output_tx.text})

        grounding = getattr(content, "grounding_metadata", None)
        if grounding is not None:
            audit = self._grounding.audit(grounding, turn_index=self._accountant.turn_index)
            await self._sink.event("citations", audit.as_event())

        if getattr(content, "interrupted", False):
            self._cancel_tool_tasks()
            await self._sink.event("interrupted", {})
            await self._finalize_turn()

        if getattr(content, "turn_complete", False):
            await self._finalize_turn()

    # ------------------------------------------------------------------ tools

    async def _handle_tool_call(self, tool_call: Any) -> None:
        calls = list(getattr(tool_call, "function_calls", None) or [])
        if not calls:
            return
        names = [getattr(c, "name", "?") for c in calls]
        call_ids = [str(getattr(c, "id", "")) for c in calls if getattr(c, "id", None)]
        self._pending_tools.update(call_ids)
        await self._sink.event("tool_call", {"names": names})

        task = asyncio.create_task(self._execute_tool_calls(calls, names, call_ids))
        for cid in call_ids or ["_anon"]:
            self._tool_tasks[cid] = task

    async def _execute_tool_calls(
        self, calls: list[Any], names: list[str], call_ids: list[str]
    ) -> None:
        try:
            responses = await self._registry.dispatch_all(calls)
            # Filter out responses for any call IDs that were cancelled while running.
            if call_ids:
                responses = [
                    r
                    for r in responses
                    if getattr(r, "id", None) is None
                    or str(getattr(r, "id", "")) in self._pending_tools
                ]
                self._pending_tools.difference_update(call_ids)
            if responses and self._session is not None:
                await self._session.send_tool_response(function_responses=responses)
            if responses:
                await self._sink.event(
                    "tool_result",
                    {
                        "names": [getattr(r, "name", "?") for r in responses],
                        "errors": [
                            r.name
                            for r in responses
                            if isinstance(getattr(r, "response", None), dict)
                            and "error" in r.response
                        ],
                    },
                )
        except asyncio.CancelledError:
            self._pending_tools.difference_update(call_ids)
            raise
        finally:
            for cid in call_ids or ["_anon"]:
                self._tool_tasks.pop(cid, None)

    async def _wait_tool_tasks(self) -> None:
        tasks = list(set(self._tool_tasks.values()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _cancel_tool_tasks(self) -> None:
        self._pending_tools.clear()
        for t in set(self._tool_tasks.values()):
            if not t.done():
                t.cancel()
        self._tool_tasks.clear()

    # ------------------------------------------------------------------ usage

    async def _finalize_turn(self) -> None:
        if not self._accountant.has_inflight:
            return
        turn = self._accountant.commit_turn()
        if turn is None:
            return

        growth = self._cost.observe_turn(turn)
        session_total = self._accountant.session_total()

        if self._cfg.usage.show_in_ui:
            payload = turn.as_event()
            payload["cost"] = growth.as_event()
            estimate = self._cost.estimate_cost(turn)
            if estimate is not None:
                payload["cost"]["estimated_cost"] = round(estimate, 6)
            await self._sink.event("usage_turn", payload)
            await self._sink.event("usage_session", session_total.as_event())

        for alert in self._cost.check_alerts(turn, session_total.total_tokens):
            await self._sink.event(
                "usage_alert", {"kind": alert.kind, "message": alert.message}
            )

    @property
    def accountant(self) -> TokenAccountant:
        return self._accountant

    @property
    def cost_model(self) -> CostModel:
        return self._cost

    async def close(self) -> None:
        self._closed = True
        await self.uplink.close()
