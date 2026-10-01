"""The Live session runner: uplink/downlink pumps, tool loop, token accounting."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..search.domain_policy import DomainPolicy
from ..search.grounding_filter import GroundingFilter
from ..settings.schema import AppConfig
from ..telemetry import mark_session_input_timestamp, sync_turn_latency_to_telemetry
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
    # output model" or "Unsupported modality for Vertex Live API".
    "is not supported",
    "not supported for",
    "unsupported modality",
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

    async def client_content(
        self,
        text: str,
        *,
        role: str = "user",
        turn_complete: bool = True,
    ) -> None:
        await self.queue.put(
            {
                "kind": "client_content",
                "text": text,
                "role": role,
                "turn_complete": turn_complete,
            }
        )

    async def mic_off(self) -> None:
        await self.queue.put({"kind": "audio_stream_end"})

    async def activity_start(self) -> None:
        await self.queue.put({"kind": "activity_start"})

    async def activity_end(self) -> None:
        await self.queue.put({"kind": "activity_end"})

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
        self._last_consumed_client_message_index: int | None = None
        self._closed = False
        self._received_any = False
        self._pending_tools: set[str] = set()
        self._inflight_tool_signatures: dict[str, tuple[str, str]] = {}
        self._tool_tasks: dict[str, asyncio.Task[Any]] = {}
        self._interaction_status: str | None = None

        # Per-turn latency tracking (synced with gemini-live-telemetry)
        self._setup_latency_ms: float | None = None
        self._last_user_input_ts: float | None = None
        self._turn_first_byte_ts: float | None = None
        self._turn_ttfb_ms: float | None = None
        self._turn_interrupted: bool = False
        self._turn_tool_rtt_ms: list[float] = []

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
                    t_connect = time.monotonic()
                    async with self._client.aio.live.connect(
                        model=self._cfg.model.name, config=config
                    ) as session:
                        self._setup_latency_ms = round(
                            (time.monotonic() - t_connect) * 1000.0, 2
                        )
                        self._session = session
                        await self._sink.event(
                            "connected",
                            {
                                "model": self._cfg.model.name,
                                "tools": self._registry.names,
                                "resumed": self._resume_handle is not None,
                                "language": self._cfg.speech.language_code,
                                "language_mode": self._cfg.speech.language_mode,
                                "setup_latency_ms": self._setup_latency_ms,
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
                if self._turn_first_byte_ts is None and self._last_user_input_ts is None:
                    self._last_user_input_ts = time.monotonic()
                await session.send_realtime_input(
                    audio=types.Blob(data=msg["data"], mime_type=INPUT_MIME)
                )
            elif kind == "text":
                # send_realtime_input for live streaming user text input.
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(session, now)
                await session.send_realtime_input(text=msg["text"])
            elif kind == "client_content":
                # Full-session client content updates with explicit role (user | model)
                # supported across the session lifecycle on Gemini 3.8 Live models.
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(session, now)
                role = str(msg.get("role") or "user")
                turn_complete = bool(msg.get("turn_complete", True))
                send_fn = getattr(session, "send_client_content", None)
                if send_fn is not None:
                    await send_fn(
                        turns=types.Content(
                            role=role,
                            parts=[types.Part(text=str(msg.get("text", "")))],
                        ),
                        turn_complete=turn_complete,
                    )
            elif kind == "audio_stream_end":
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(session, now)
                await session.send_realtime_input(audio_stream_end=True)
            elif kind == "activity_start":
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(session, now)
                await session.send_realtime_input(
                    activity_start=types.ActivityStart()
                )
            elif kind == "activity_end":
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(session, now)
                await session.send_realtime_input(
                    activity_end=types.ActivityEnd()
                )

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
                self._inflight_tool_signatures.pop(cid, None)
                t = self._tool_tasks.pop(cid, None)
                if t is not None and not t.done():
                    t.cancel()
            await self._sink.event("tool_cancelled", {"ids": ids})

        update = getattr(message, "session_resumption_update", None)
        if update is not None:
            last_idx = getattr(update, "last_consumed_client_message_index", None)
            if last_idx is not None:
                self._last_consumed_client_message_index = int(last_idx)
            if getattr(update, "resumable", False):
                self._resume_handle = getattr(update, "new_handle", None)

        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            await self._sink.event(
                "go_away", {"time_left": str(getattr(go_away, "time_left", ""))}
            )

        voice_activity = getattr(message, "voice_activity", None)
        vad_signal = getattr(message, "voice_activity_detection_signal", None)
        if voice_activity is not None or vad_signal is not None:
            va_type = (
                getattr(voice_activity, "voice_activity_type", None)
                if voice_activity is not None
                else None
            )
            sig_type = (
                getattr(vad_signal, "vad_signal_type", None)
                if vad_signal is not None
                else None
            )
            await self._sink.event(
                "voice_activity",
                {
                    "voice_activity_type": (
                        str(getattr(va_type, "value", va_type))
                        if va_type is not None
                        else None
                    ),
                    "vad_signal_type": (
                        str(getattr(sig_type, "value", sig_type))
                        if sig_type is not None
                        else None
                    ),
                },
            )

        # Gemini 3.8 Live Extended Thinking emits interaction_status ("IN_PROGRESS" | "IDLE")
        # to signal whether background reasoning / async tools have completed.
        interaction_status = getattr(message, "interaction_status", None) or getattr(
            message, "interactionStatus", None
        )
        if interaction_status is not None:
            status_val = str(getattr(interaction_status, "value", interaction_status))
            self._interaction_status = status_val
            await self._sink.event("interaction_status", {"status": status_val})
            if status_val == "IDLE" and self._accountant.has_inflight:
                await self._finalize_turn()

    def _mark_first_model_byte(self) -> None:
        if self._turn_first_byte_ts is None:
            now = time.monotonic()
            self._turn_first_byte_ts = now
            if self._last_user_input_ts is not None:
                self._turn_ttfb_ms = round(
                    max(1.0, (now - self._last_user_input_ts) * 1000.0), 2
                )

    async def _handle_server_content(self, content: Any) -> None:
        # A single event can carry multiple parts (audio AND transcript), so we
        # must process all of them rather than branching on the first match.
        input_tx = getattr(content, "input_transcription", None)
        if input_tx is not None and getattr(input_tx, "text", None):
            if self._turn_first_byte_ts is None:
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(self._session, now)
            await self._sink.event("transcript", {"role": "user", "text": input_tx.text})

        model_turn = getattr(content, "model_turn", None)
        if model_turn is not None:
            for part in getattr(model_turn, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    self._mark_first_model_byte()
                    await self._sink.audio(inline.data)
                text = getattr(part, "text", None)
                if text:
                    self._mark_first_model_byte()
                    await self._sink.event("text", {"text": text})

        output_tx = getattr(content, "output_transcription", None)
        if output_tx is not None and getattr(output_tx, "text", None):
            self._mark_first_model_byte()
            await self._sink.event("transcript", {"role": "model", "text": output_tx.text})

        grounding = getattr(content, "grounding_metadata", None)
        if grounding is not None:
            audit = self._grounding.audit(grounding, turn_index=self._accountant.turn_index)
            await self._sink.event("citations", audit.as_event())

        if getattr(content, "interrupted", False):
            self._turn_interrupted = True
            self._cancel_tool_tasks()
            await self._sink.event("interrupted", {})
            await self._finalize_turn()

        if getattr(content, "turn_complete", False):
            await self._finalize_turn()

    # ------------------------------------------------------------------ tools

    async def _handle_tool_call(self, tool_call: Any) -> None:
        import json

        calls = list(getattr(tool_call, "function_calls", None) or [])
        if not calls:
            return
        self._mark_first_model_byte()

        # Deduplicate identical in-flight asynchronous tool calls (Gemini 3.8 Live
        # best practice: when a user repeats a request while a NON_BLOCKING tool is
        # still running, prevent duplicate execution while acknowledging the call ID).
        active_sigs = set(self._inflight_tool_signatures.values())
        unique_calls: list[Any] = []
        duplicate_calls: list[Any] = []
        batch_sigs: dict[str, tuple[str, str]] = {}

        for c in calls:
            cname = str(getattr(c, "name", "?") or "?")
            cargs = dict(getattr(c, "args", None) or {})
            cid = str(getattr(c, "id", "") or "")
            sig = (cname, json.dumps(cargs, sort_keys=True, default=str))
            if sig in active_sigs or sig in batch_sigs.values():
                duplicate_calls.append(c)
            else:
                unique_calls.append(c)
                if cid:
                    batch_sigs[cid] = sig

        if duplicate_calls and self._session is not None:
            from google.genai import types

            dup_responses = [
                types.FunctionResponse(
                    id=getattr(dc, "id", None),
                    name=str(getattr(dc, "name", "?") or "?"),
                    response={
                        "status": "duplicate_in_flight",
                        "retryable": False,
                        "message": (
                            f"Tool '{getattr(dc, 'name', '?')}' with identical arguments "
                            "is already executing in flight. Wait for its result."
                        ),
                    },
                )
                for dc in duplicate_calls
            ]
            await self._session.send_tool_response(function_responses=dup_responses)
            await self._sink.event(
                "tool_duplicate_skipped",
                {"names": [getattr(dc, "name", "?") for dc in duplicate_calls]},
            )

        if not unique_calls:
            return

        names = [getattr(c, "name", "?") for c in unique_calls]
        call_ids = [
            str(getattr(c, "id", "")) for c in unique_calls if getattr(c, "id", None)
        ]
        call_details = [
            {
                "id": str(getattr(c, "id", "") or ""),
                "name": getattr(c, "name", "?"),
                "args": dict(getattr(c, "args", None) or {}),
            }
            for c in unique_calls
        ]
        for cd in call_details:
            log.info("Tool call -> %s(%s)", cd["name"], cd["args"])

        self._pending_tools.update(call_ids)
        self._inflight_tool_signatures.update(batch_sigs)
        await self._sink.event(
            "tool_call",
            {"names": names, "calls": call_details},
        )

        task = asyncio.create_task(self._execute_tool_calls(unique_calls, names, call_ids))
        for cid in call_ids or ["_anon"]:
            self._tool_tasks[cid] = task

    async def _execute_tool_calls(
        self, calls: list[Any], names: list[str], call_ids: list[str]
    ) -> None:
        t0 = time.monotonic()
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
            rtt_ms = round((time.monotonic() - t0) * 1000.0, 2)
            self._turn_tool_rtt_ms.append(rtt_ms)
            # If the model hasn't started speaking its final answer yet, measure
            # post-tool TTFB from when the tool response was sent.
            if self._turn_first_byte_ts is None:
                now = time.monotonic()
                self._last_user_input_ts = now
                mark_session_input_timestamp(self._session, now)
            if responses:
                result_details = [
                    {
                        "id": str(getattr(r, "id", "") or ""),
                        "name": getattr(r, "name", "?"),
                        "response": getattr(r, "response", None),
                    }
                    for r in responses
                ]
                for rd in result_details:
                    log.info(
                        "Tool result <- %s (%.1fms): %s", rd["name"], rtt_ms, rd["response"]
                    )
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
                        "results": result_details,
                        "round_trip_ms": rtt_ms,
                    },
                )
        except asyncio.CancelledError:
            self._pending_tools.difference_update(call_ids)
            raise
        finally:
            for cid in call_ids or ["_anon"]:
                self._inflight_tool_signatures.pop(cid, None)
                self._tool_tasks.pop(cid, None)

    async def _wait_tool_tasks(self) -> None:
        tasks = list(set(self._tool_tasks.values()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _cancel_tool_tasks(self) -> None:
        self._pending_tools.clear()
        self._inflight_tool_signatures.clear()
        for t in set(self._tool_tasks.values()):
            if not t.done():
                t.cancel()
        self._tool_tasks.clear()

    # ------------------------------------------------------------------ usage

    async def _finalize_turn(self) -> None:
        turn_duration_ms: float | None = None
        if self._turn_first_byte_ts is not None:
            turn_duration_ms = round(
                max(1.0, (time.monotonic() - self._turn_first_byte_ts) * 1000.0), 2
            )
        ttfb_ms = self._turn_ttfb_ms
        was_interrupted = self._turn_interrupted
        avg_tool_rtt = (
            round(sum(self._turn_tool_rtt_ms) / len(self._turn_tool_rtt_ms), 2)
            if self._turn_tool_rtt_ms
            else None
        )

        # Reset per-turn latency trackers for the next turn
        self._turn_first_byte_ts = None
        self._turn_ttfb_ms = None
        self._last_user_input_ts = None
        self._turn_interrupted = False
        self._turn_tool_rtt_ms.clear()

        if not self._accountant.has_inflight:
            return
        turn = self._accountant.commit_turn()
        if turn is None:
            return

        sync_turn_latency_to_telemetry(
            self._session,
            turn_index=turn.turn_index,
            ttfb_ms=ttfb_ms,
            turn_duration_ms=turn_duration_ms,
            was_interrupted=was_interrupted,
        )

        growth = self._cost.observe_turn(turn)
        session_total = self._accountant.session_total()

        if self._cfg.usage.show_in_ui:
            payload = turn.as_event()
            payload["cost"] = growth.as_event()
            estimate = self._cost.estimate_cost(turn)
            if estimate is not None:
                payload["cost"]["estimated_cost"] = round(estimate, 6)
            payload["latency"] = {
                "ttfb_ms": ttfb_ms,
                "turn_duration_ms": turn_duration_ms,
                "setup_latency_ms": self._setup_latency_ms,
                "tool_round_trip_ms": avg_tool_rtt,
                "was_interrupted": was_interrupted,
            }
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

    @property
    def interaction_status(self) -> str | None:
        return self._interaction_status

    @property
    def last_consumed_client_message_index(self) -> int | None:
        return self._last_consumed_client_message_index

    async def close(self) -> None:
        self._closed = True
        await self.uplink.close()
