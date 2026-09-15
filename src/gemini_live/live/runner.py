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
    "invalid_argument",
    "failed_precondition",
    "has not been used in project",
    "api has not been used",
    "is disabled",
    "was not found",
    "not found for api version",
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

    Config and credential failures surface as a WebSocket close rather than a
    clean exception type, so this has to inspect the message text.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_MARKERS)


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
        self._cost = cost_model or CostModel(
            warn_prompt_tokens_per_turn=cfg.usage.alerts.warn_prompt_tokens_per_turn,
            warn_rent_ratio=cfg.usage.alerts.warn_rent_ratio,
            warn_session_total=cfg.usage.alerts.warn_session_total,
        )

        self.uplink = _Uplink(asyncio.Queue())
        self._session: Any | None = None
        self._resume_handle: str | None = None
        self._closed = False
        self._pending_tools: set[str] = set()

    # ------------------------------------------------------------------- run

    async def run(self) -> None:
        """Connect and pump until closed. Reconnects transparently when the
        server sends GoAway and session resumption is enabled."""
        attempts = 0
        while not self._closed:
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
                    attempts = 0
                    await self._sink.event(
                        "connected",
                        {
                            "model": self._cfg.model.name,
                            "tools": self._registry.names,
                            "resumed": self._resume_handle is not None,
                        },
                    )
                    await self._pump(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
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
                attempts += 1
                if not self._cfg.session.resumption.enabled or attempts > 3:
                    await self._sink.event("error", {"message": f"{type(exc).__name__}: {exc}"})
                    raise
                delay = min(2**attempts, 8)
                log.warning("live session dropped (%s); reconnecting in %ss", exc, delay)
                await self._sink.event("reconnecting", {"in_seconds": delay, "reason": str(exc)})
                await asyncio.sleep(delay)
            finally:
                self._session = None

        # A call that ends mid-generation should still be billed.
        await self._finalize_turn()

    async def _pump(self, session: Any) -> None:
        async with asyncio.TaskGroup() as tg:
            up = tg.create_task(self._pump_uplink(session))
            down = tg.create_task(self._pump_downlink(session))
            done, pending = await asyncio.wait(
                {up, down}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()

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
        async for message in session.receive():
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
                ids = list(getattr(cancellation, "ids", None) or [])
                self._pending_tools.difference_update(ids)
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
            await self._sink.event("interrupted", {})

        if getattr(content, "turn_complete", False):
            await self._finalize_turn()

    # ------------------------------------------------------------------ tools

    async def _handle_tool_call(self, tool_call: Any) -> None:
        calls = list(getattr(tool_call, "function_calls", None) or [])
        if not calls:
            return
        names = [getattr(c, "name", "?") for c in calls]
        self._pending_tools.update(
            str(getattr(c, "id", "")) for c in calls if getattr(c, "id", None)
        )
        await self._sink.event("tool_call", {"names": names})

        responses = await self._registry.dispatch_all(calls)
        if self._session is not None:
            await self._session.send_tool_response(function_responses=responses)
        await self._sink.event(
            "tool_result",
            {
                "names": names,
                "errors": [
                    r.name
                    for r in responses
                    if isinstance(getattr(r, "response", None), dict)
                    and "error" in r.response
                ],
            },
        )

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
