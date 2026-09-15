"""Data models for token accounting.

Important API shape note, verified against ``google/genai/types.py``:

The Live path uses ``types.UsageMetadata`` (NOT
``GenerateContentResponseUsageMetadata``), so the response counter is named
``response_token_count``, not ``candidates_token_count``.

There are exactly FOUR per-modality ``ModalityTokenCount`` arrays:

    prompt_tokens_details, cache_tokens_details,
    response_tokens_details, tool_use_prompt_tokens_details

There is no ``thoughts_tokens_details`` and no ``total_tokens_details``. Thinking
tokens are therefore a scalar only, and any "total by modality" figure is derived
by summing the four real arrays -- we label it as derived so it is never mistaken
for a server-reported number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ModalityMap = dict[str, int]

UNKNOWN_MODALITY = "UNKNOWN"
"""Bucket for ModalityTokenCount entries whose ``modality`` is None.

Recorded fixtures really do contain these. Dropping them would under-report;
crashing would be worse.
"""


def merge_modalities(*maps: ModalityMap) -> ModalityMap:
    out: ModalityMap = {}
    for m in maps:
        for key, value in m.items():
            out[key] = out.get(key, 0) + value
    return out


@dataclass(frozen=True)
class TurnUsage:
    """Committed usage for one conversational turn."""

    turn_index: int

    # --- the six scalar counters ---
    prompt_tokens: int = 0
    cached_tokens: int = 0
    response_tokens: int = 0
    tool_use_prompt_tokens: int = 0
    thoughts_tokens: int = 0  # scalar only: no per-modality breakdown exists
    total_tokens: int = 0

    # --- the four real per-modality breakdowns ---
    prompt_by_modality: ModalityMap = field(default_factory=dict)
    cache_by_modality: ModalityMap = field(default_factory=dict)
    response_by_modality: ModalityMap = field(default_factory=dict)
    tool_use_by_modality: ModalityMap = field(default_factory=dict)

    # --- provenance / diagnostics ---
    traffic_type: str | None = None
    service_tier: str | None = None  # always None on Vertex; captured anyway
    accounting_mode: str = "last_wins"
    response_tokens_delta_sum: int = 0
    snapshots: int = 0

    @property
    def total_by_modality(self) -> ModalityMap:
        """DERIVED: sum of the four real arrays. Not reported by the API."""
        return merge_modalities(
            self.prompt_by_modality,
            self.cache_by_modality,
            self.response_by_modality,
            self.tool_use_by_modality,
        )

    @property
    def context_rent(self) -> int:
        """Tokens billed this turn purely to re-read the resident context.

        Every turn is charged for the entire context -- history plus system
        instruction plus tool declarations -- not just the new input.
        """
        return self.prompt_tokens

    @property
    def marginal_tokens(self) -> int:
        """Tokens representing genuinely new work this turn."""
        return self.response_tokens + self.thoughts_tokens + self.tool_use_prompt_tokens

    @property
    def rent_ratio(self) -> float:
        """Fraction of this turn's bill spent re-reading context. Approaching 1.0
        means you are paying mostly to re-send history."""
        if not self.total_tokens:
            return 0.0
        return self.context_rent / self.total_tokens

    def as_event(self, *, include_service_tier: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "turn": self.turn_index,
            "scalars": {
                "prompt": self.prompt_tokens,
                "cached": self.cached_tokens,
                "response": self.response_tokens,
                "tool_use": self.tool_use_prompt_tokens,
                "thoughts": self.thoughts_tokens,
                "total": self.total_tokens,
            },
            "by_modality": {
                "prompt": self.prompt_by_modality,
                "cache": self.cache_by_modality,
                "response": self.response_by_modality,
                "tool_use": self.tool_use_by_modality,
                "total_derived": self.total_by_modality,
            },
            "traffic_type": self.traffic_type,
            "accounting_mode": self.accounting_mode,
            "response_tokens_delta_sum": self.response_tokens_delta_sum,
            "snapshots": self.snapshots,
        }
        if include_service_tier:
            payload["service_tier"] = self.service_tier
        return payload


@dataclass(frozen=True)
class SessionUsage:
    turns: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    response_tokens: int = 0
    tool_use_prompt_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    by_modality: ModalityMap = field(default_factory=dict)

    def as_event(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "prompt": self.prompt_tokens,
            "cached": self.cached_tokens,
            "response": self.response_tokens,
            "tool_use": self.tool_use_prompt_tokens,
            "thoughts": self.thoughts_tokens,
            "total": self.total_tokens,
            "by_modality": self.by_modality,
        }


@dataclass(frozen=True)
class ContextGrowth:
    """How the billed context changed on a given turn."""

    turn_index: int
    prompt_tokens: int
    delta: int
    compression_event: bool
    rent_ratio: float
    tool_declaration_tokens: int
    projected_tool_cost: int
    """Tool declarations re-billed every turn, cumulative so far."""

    def as_event(self) -> dict[str, Any]:
        return {
            "context_rent": self.prompt_tokens,
            "prompt_delta": self.delta,
            "compression_event": self.compression_event,
            "rent_ratio": round(self.rent_ratio, 4),
            "tool_declaration_tokens": self.tool_declaration_tokens,
            "projected_tool_cost": self.projected_tool_cost,
        }
