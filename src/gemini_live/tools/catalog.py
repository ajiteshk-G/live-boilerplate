"""Tool catalog: the full, uncurated set of candidate tools."""

from __future__ import annotations

import fnmatch
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .schema_utils import normalize_schema, truncate_description

ToolInvoker = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolCandidate:
    """One tool that *could* be offered to the model."""

    exposed_name: str
    """Name as the model will see it: ``<server>__<tool>`` or a bare builtin name."""

    description: str
    input_schema: dict[str, Any]
    invoke: ToolInvoker
    origin: str
    """``mcp:<server>`` or ``builtin``."""

    raw_name: str = ""
    token_cost: int = 0
    score: float = 0.0
    pinned: bool = False
    drop_reason: str | None = None
    embedding: list[float] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Text used for semantic ranking."""
        params = ", ".join(sorted((self.input_schema.get("properties") or {}).keys()))
        base = f"{self.exposed_name}: {self.description}"
        return f"{base} (parameters: {params})" if params else base

    def compacted(self, max_desc_chars: int) -> ToolCandidate:
        self.description = truncate_description(self.description, max_desc_chars)
        self.input_schema = normalize_schema(self.input_schema)
        return self


def matches_any(name: str, patterns: list[str]) -> bool:
    """Glob match, so config can say ``*__delete_*``."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


@dataclass
class Selection:
    """Result of curation."""

    selected: list[ToolCandidate]
    dropped: list[ToolCandidate]
    total_tokens: int
    catalog_size: int
    catalog_tokens: int
    mode: str

    @property
    def saved_tokens(self) -> int:
        return self.catalog_tokens - self.total_tokens

    def projected_cost(self, turns: int) -> int:
        """Declarations are re-billed every turn, so this is what the selection
        actually costs over a conversation of ``turns`` turns."""
        return self.total_tokens * turns
