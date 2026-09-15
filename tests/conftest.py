"""Shared test helpers.

The whole suite runs offline: no GCP credentials, no network, no MCP servers.
Anything that would call a service is replaced by a fake here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# --------------------------------------------------------------------- usage


@dataclass
class FakeModality:
    """Stand-in for ``types.ModalityTokenCount``.

    ``modality`` is deliberately allowed to be None: real recorded fixtures
    contain entries like that, and the accountant must not drop or crash on them.
    """

    modality: str | None
    token_count: int


@dataclass
class FakeUsage:
    """Stand-in for ``types.UsageMetadata`` (the Live variant).

    Note the field is ``response_token_count``. The Live path does NOT use
    ``candidates_token_count`` -- that belongs to the non-Live
    ``GenerateContentResponseUsageMetadata``.
    """

    prompt_token_count: int = 0
    cached_content_token_count: int = 0
    response_token_count: int = 0
    tool_use_prompt_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0

    prompt_tokens_details: list[FakeModality] = field(default_factory=list)
    cache_tokens_details: list[FakeModality] = field(default_factory=list)
    response_tokens_details: list[FakeModality] = field(default_factory=list)
    tool_use_prompt_tokens_details: list[FakeModality] = field(default_factory=list)

    traffic_type: Any = None
    service_tier: Any = None


def cumulative_turn(
    prompt: int,
    response_steps: list[int],
    *,
    thoughts: int = 0,
    tool_use: int = 0,
) -> list[FakeUsage]:
    """Build the snapshot sequence for one turn of a CUMULATIVE stream.

    ``response_steps`` are running totals, e.g. [10, 25, 40] means the server
    said "40 response tokens so far" by the end -- not 75.
    """
    return [
        FakeUsage(
            prompt_token_count=prompt,
            response_token_count=r,
            thoughts_token_count=thoughts,
            tool_use_prompt_token_count=tool_use,
            total_token_count=prompt + r + thoughts + tool_use,
            response_tokens_details=[FakeModality("AUDIO", r)],
            prompt_tokens_details=[FakeModality("AUDIO", prompt)],
        )
        for r in response_steps
    ]


def delta_turn(prompt: int, response_deltas: list[int]) -> list[FakeUsage]:
    """Build the snapshot sequence for one turn of a DELTA stream.

    Each message reports only the tokens produced since the last one, so the
    counter is NOT monotonic across messages.
    """
    running = 0
    out: list[FakeUsage] = []
    for d in response_deltas:
        running += d
        out.append(
            FakeUsage(
                prompt_token_count=prompt,
                response_token_count=d,
                total_token_count=prompt + running,
                response_tokens_details=[FakeModality("AUDIO", d)],
            )
        )
    return out


# ---------------------------------------------------------------------- tools


@pytest.fixture
def make_candidate():
    """Factory for ToolCandidate without needing an MCP server."""
    from gemini_live.tools.catalog import ToolCandidate

    def _make(
        name: str,
        *,
        description: str = "does a thing",
        cost: int = 100,
        schema: dict[str, Any] | None = None,
        origin: str = "mcp:test",
        embedding: list[float] | None = None,
    ) -> ToolCandidate:
        async def _invoke(_args: dict[str, Any]) -> dict[str, Any]:
            return {"ok": name}

        cand = ToolCandidate(
            exposed_name=name,
            description=description,
            input_schema=schema or {"type": "object", "properties": {}},
            invoke=_invoke,
            origin=origin,
            token_cost=cost,
        )
        if embedding:
            cand.embedding = embedding
        return cand

    return _make


class StubEmbedder:
    """Embedder returning fixed vectors, so ranking is deterministic.

    Unknown text gets an orthogonal vector, i.e. similarity 0.
    """

    def __init__(self, vectors: dict[str, list[float]], *, fail: bool = False) -> None:
        self._vectors = vectors
        self._fail = fail
        self.calls = 0

    def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        self.calls += 1
        if self._fail:
            raise RuntimeError("embedding backend unavailable")
        out = []
        for text in texts:
            match = next((v for k, v in self._vectors.items() if k in text), None)
            out.append(match or [0.0, 0.0, 1.0])
        return out
