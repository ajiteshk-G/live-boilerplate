"""Turn-boundary token accounting for the Live API.

THE CRITICAL SEMANTIC
---------------------
``usageMetadata`` on the Live API is **cumulative within a turn** and **resets at
each ``turnComplete``**. It is neither a per-message delta for the whole session
nor a running session total.

    "UsageMetadata is cumulative-per-turn, not per-session: the count resets
     each turn_complete."
    -- prototypes/projects/liveapi/thinker_talker/usage_metrics/live_api_usage.py

So the correct algorithm is:

    last-wins within a turn  ->  commit on turnComplete  ->  sum committed turns

Naively doing ``total += msg.usage_metadata.total_token_count`` for every message
massively over-counts, because most messages restate a running total.

THE UNRESOLVED NUANCE
---------------------
A second internal source (sherlog_token_stats/bidi_usage.py) reports that on the
raw wire the *response* counters behave as per-message deltas while prompt/total
are running totals. Both sources agree prompt/total must not be summed per
message, but they disagree about the response counters.

Rather than guess, this accountant tracks BOTH interpretations simultaneously and
``mode="auto"`` selects based on observed monotonicity. Every raw snapshot is
logged so ``glive calibrate-usage`` can settle the question empirically.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Literal

from .models import UNKNOWN_MODALITY, ModalityMap, SessionUsage, TurnUsage, merge_modalities

log = logging.getLogger(__name__)

AccountingMode = Literal["auto", "last_wins", "delta_sum"]


def _modality_map(details: Any) -> ModalityMap:
    """Convert a list[ModalityTokenCount] into {modality: tokens}.

    Entries with ``modality=None`` are bucketed under UNKNOWN rather than dropped;
    real recorded fixtures contain them.
    """
    out: ModalityMap = {}
    for item in details or []:
        modality = getattr(item, "modality", None)
        name = getattr(modality, "value", modality) or UNKNOWN_MODALITY
        count = getattr(item, "token_count", None) or 0
        out[str(name)] = out.get(str(name), 0) + int(count)
    return out


def _enum_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _i(value: Any) -> int:
    return int(value or 0)


class TokenAccountant:
    """Accumulates ``usage_metadata`` snapshots into per-turn and session totals."""

    def __init__(
        self,
        *,
        mode: AccountingMode = "auto",
        raw_log_path: str | Path | None = None,
        log_raw_snapshots: bool = True,
        session_id: str | None = None,
    ) -> None:
        self._mode: AccountingMode = mode
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._turn_index = 0
        self._turns: list[TurnUsage] = []

        # in-flight turn state
        self._inflight: Any | None = None
        self._snapshots = 0
        self._delta_response = 0
        self._monotonic = True
        self._raw_this_turn: list[dict[str, Any]] = []

        self._raw_log_path = Path(raw_log_path) if raw_log_path else None
        self._log_raw = log_raw_snapshots
        if self._raw_log_path:
            self._raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ observe

    def observe(self, um: Any) -> None:
        """Record one ``usage_metadata`` snapshot. Safe to call on any message."""
        if um is None:
            return

        self._snapshots += 1
        cur = _i(getattr(um, "response_token_count", None))
        prev = _i(getattr(self._inflight, "response_token_count", None)) if self._inflight else None

        # The two competing hypotheses are tracked independently, because we
        # cannot tell from a single message which one is true:
        #
        #   last_wins -- each message restates a running total  -> final value
        #   delta_sum -- each message reports only new tokens   -> sum of values
        #
        # So delta_sum must be a plain sum. Subtracting consecutive values here
        # would silently re-implement the last_wins hypothesis and under-count a
        # genuine delta stream.
        self._delta_response += cur

        # A counter that goes backwards cannot be a running total. That is the
        # only unambiguous evidence available in-band.
        if prev is not None and cur < prev:
            self._monotonic = False

        self._inflight = um

        if self._log_raw:
            self._raw_this_turn.append(self._snapshot_dict(um))

    def _snapshot_dict(self, um: Any) -> dict[str, Any]:
        return {
            "kind": "snapshot",
            "session_id": self._session_id,
            "turn": self._turn_index,
            "seq": self._snapshots,
            "prompt_token_count": _i(getattr(um, "prompt_token_count", None)),
            "cached_content_token_count": _i(getattr(um, "cached_content_token_count", None)),
            "response_token_count": _i(getattr(um, "response_token_count", None)),
            "tool_use_prompt_token_count": _i(getattr(um, "tool_use_prompt_token_count", None)),
            "thoughts_token_count": _i(getattr(um, "thoughts_token_count", None)),
            "total_token_count": _i(getattr(um, "total_token_count", None)),
            "prompt_tokens_details": _modality_map(getattr(um, "prompt_tokens_details", None)),
            "cache_tokens_details": _modality_map(getattr(um, "cache_tokens_details", None)),
            "response_tokens_details": _modality_map(getattr(um, "response_tokens_details", None)),
            "tool_use_prompt_tokens_details": _modality_map(
                getattr(um, "tool_use_prompt_tokens_details", None)
            ),
            "traffic_type": _enum_str(getattr(um, "traffic_type", None)),
            "service_tier": _enum_str(getattr(um, "service_tier", None)),
        }

    # ------------------------------------------------------------------- commit

    @property
    def has_inflight(self) -> bool:
        return self._inflight is not None

    @property
    def turn_index(self) -> int:
        """Index of the turn currently being accumulated."""
        return self._turn_index

    def commit_turn(self) -> TurnUsage | None:
        """Finalise the current turn. Call on ``turnComplete`` AND on session close
        so a turn interrupted mid-generation is still billed rather than lost."""
        um = self._inflight
        if um is None:
            return None

        if self._mode == "auto":
            resolved = "last_wins" if self._monotonic else "delta_sum"
        else:
            resolved = self._mode

        last_wins_response = _i(getattr(um, "response_token_count", None))
        response = last_wins_response if resolved == "last_wins" else self._delta_response

        turn = TurnUsage(
            turn_index=self._turn_index,
            prompt_tokens=_i(getattr(um, "prompt_token_count", None)),
            cached_tokens=_i(getattr(um, "cached_content_token_count", None)),
            response_tokens=response,
            tool_use_prompt_tokens=_i(getattr(um, "tool_use_prompt_token_count", None)),
            thoughts_tokens=_i(getattr(um, "thoughts_token_count", None)),
            total_tokens=_i(getattr(um, "total_token_count", None)),
            prompt_by_modality=_modality_map(getattr(um, "prompt_tokens_details", None)),
            cache_by_modality=_modality_map(getattr(um, "cache_tokens_details", None)),
            response_by_modality=_modality_map(getattr(um, "response_tokens_details", None)),
            tool_use_by_modality=_modality_map(
                getattr(um, "tool_use_prompt_tokens_details", None)
            ),
            traffic_type=_enum_str(getattr(um, "traffic_type", None)),
            service_tier=_enum_str(getattr(um, "service_tier", None)),
            accounting_mode=resolved,
            response_tokens_delta_sum=self._delta_response,
            snapshots=self._snapshots,
        )
        self._turns.append(turn)
        self._write_turn(turn)
        self._reset_turn()
        return turn

    def _reset_turn(self) -> None:
        self._turn_index += 1
        self._inflight = None
        self._snapshots = 0
        self._delta_response = 0
        self._monotonic = True
        self._raw_this_turn = []

    def _write_turn(self, turn: TurnUsage) -> None:
        if not self._raw_log_path:
            return
        try:
            with self._raw_log_path.open("a") as fh:
                for snap in self._raw_this_turn:
                    fh.write(json.dumps(snap) + "\n")
                record = {
                    "kind": "turn",
                    "session_id": self._session_id,
                    **turn.as_event(include_service_tier=True),
                }
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # logging must never break the call
            log.warning("could not write usage log: %s", exc)

    # ------------------------------------------------------------------ session

    @property
    def turns(self) -> list[TurnUsage]:
        return list(self._turns)

    def session_total(self) -> SessionUsage:
        """Sum of committed per-turn snapshots."""
        if not self._turns:
            return SessionUsage()
        return SessionUsage(
            turns=len(self._turns),
            prompt_tokens=sum(t.prompt_tokens for t in self._turns),
            cached_tokens=sum(t.cached_tokens for t in self._turns),
            response_tokens=sum(t.response_tokens for t in self._turns),
            tool_use_prompt_tokens=sum(t.tool_use_prompt_tokens for t in self._turns),
            thoughts_tokens=sum(t.thoughts_tokens for t in self._turns),
            total_tokens=sum(t.total_tokens for t in self._turns),
            by_modality=merge_modalities(*(t.total_by_modality for t in self._turns)),
        )
