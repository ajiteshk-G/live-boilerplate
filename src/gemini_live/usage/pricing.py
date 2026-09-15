"""Optional token -> currency estimation.

Deliberately isolated and disabled by default: published rates go stale, and a
confidently wrong cost figure is worse than no figure. Token counts themselves are
never affected by anything in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import TurnUsage


@dataclass(frozen=True)
class PriceTable:
    text_input_per_1k: float = 0.0
    audio_input_per_1k: float = 0.0
    text_output_per_1k: float = 0.0
    audio_output_per_1k: float = 0.0
    thinking_per_1k: float = 0.0
    currency: str = "USD"

    def estimate(self, turn: TurnUsage) -> float:
        """Best-effort cost for one turn, using the per-modality splits."""
        p = turn.prompt_by_modality
        r = turn.response_by_modality

        audio_in = p.get("AUDIO", 0)
        text_in = turn.prompt_tokens - audio_in  # everything non-audio, incl. UNKNOWN
        audio_out = r.get("AUDIO", 0)
        text_out = max(turn.response_tokens - audio_out, 0)

        return (
            audio_in / 1000 * self.audio_input_per_1k
            + text_in / 1000 * self.text_input_per_1k
            + audio_out / 1000 * self.audio_output_per_1k
            + text_out / 1000 * self.text_output_per_1k
            + turn.thoughts_tokens / 1000 * self.thinking_per_1k
        )
