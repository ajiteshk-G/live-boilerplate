"""Cost model: turns raw token counts into the numbers that explain the bill.

Every turn is charged for the entire resident context -- conversation history,
system instruction, and tool declarations -- not merely the new input. That makes
two derived figures far more informative than a raw total:

* context rent   -- promptTokenCount, re-billed every single turn
* marginal cost  -- response + thoughts + tool-use, the genuinely new work

It also means tool declarations are a *recurring* charge. A 6,000-token tool
payload across a 30-turn conversation costs ~180,000 billed prompt tokens, which
is why curation matters so much.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ContextGrowth, TurnUsage
from .pricing import PriceTable


@dataclass
class Alert:
    kind: str
    message: str
    value: float


class CostModel:
    """Tracks context growth, infers compression events, and raises budget alerts."""

    def __init__(
        self,
        *,
        tool_declaration_tokens: int = 0,
        system_instruction_tokens: int = 0,
        compression_epsilon: int = 256,
        warn_prompt_tokens_per_turn: int | None = None,
        warn_rent_ratio: float | None = None,
        warn_session_total: int | None = None,
        prices: PriceTable | None = None,
    ) -> None:
        self._tool_tokens = tool_declaration_tokens
        self._sys_tokens = system_instruction_tokens
        self._epsilon = compression_epsilon
        self._warn_prompt = warn_prompt_tokens_per_turn
        self._warn_rent = warn_rent_ratio
        self._warn_total = warn_session_total
        self._prices = prices

        self._prev_prompt: int | None = None
        self._history: list[ContextGrowth] = []
        self._fired: set[str] = set()

    @property
    def tool_declaration_tokens(self) -> int:
        return self._tool_tokens

    def set_tool_declaration_tokens(self, tokens: int) -> None:
        self._tool_tokens = tokens

    def observe_turn(self, turn: TurnUsage) -> ContextGrowth:
        delta = 0 if self._prev_prompt is None else turn.prompt_tokens - self._prev_prompt
        # The API never announces compression. A sizeable DROP in prompt tokens
        # between turns is its unambiguous fingerprint.
        compressed = self._prev_prompt is not None and delta < -self._epsilon

        growth = ContextGrowth(
            turn_index=turn.turn_index,
            prompt_tokens=turn.prompt_tokens,
            delta=delta,
            compression_event=compressed,
            rent_ratio=turn.rent_ratio,
            tool_declaration_tokens=self._tool_tokens,
            projected_tool_cost=self._tool_tokens * (turn.turn_index + 1),
        )
        self._prev_prompt = turn.prompt_tokens
        self._history.append(growth)
        return growth

    def check_alerts(self, turn: TurnUsage, session_total_tokens: int) -> list[Alert]:
        """Threshold alerts. Each kind fires at most once per session (hysteresis)."""
        alerts: list[Alert] = []

        if (
            self._warn_prompt is not None
            and turn.prompt_tokens >= self._warn_prompt
            and "prompt" not in self._fired
        ):
            self._fired.add("prompt")
            alerts.append(
                Alert(
                    "prompt_tokens",
                    f"Context rent reached {turn.prompt_tokens:,} tokens per turn "
                    f"(threshold {self._warn_prompt:,}). Every turn now re-bills this much.",
                    turn.prompt_tokens,
                )
            )

        if (
            self._warn_rent is not None
            and turn.rent_ratio >= self._warn_rent
            and "rent_ratio" not in self._fired
        ):
            self._fired.add("rent_ratio")
            alerts.append(
                Alert(
                    "rent_ratio",
                    f"{turn.rent_ratio:.0%} of this turn's tokens went to re-reading context. "
                    "Consider lowering the compression trigger or trimming tools.",
                    turn.rent_ratio,
                )
            )

        if (
            self._warn_total is not None
            and session_total_tokens >= self._warn_total
            and "session_total" not in self._fired
        ):
            self._fired.add("session_total")
            alerts.append(
                Alert(
                    "session_total",
                    f"Session total reached {session_total_tokens:,} tokens "
                    f"(threshold {self._warn_total:,}).",
                    session_total_tokens,
                )
            )

        return alerts

    def cost_drivers(self, turns: int) -> dict[str, int]:
        """Attribution of the fixed, per-turn-recurring overhead."""
        return {
            "tool_declarations_per_turn": self._tool_tokens,
            "system_instruction_per_turn": self._sys_tokens,
            "tool_declarations_cumulative": self._tool_tokens * turns,
            "system_instruction_cumulative": self._sys_tokens * turns,
        }

    @property
    def history(self) -> list[ContextGrowth]:
        return list(self._history)

    def estimate_cost(self, turn: TurnUsage) -> float | None:
        if self._prices is None:
            return None
        return self._prices.estimate(turn)
