"""Intelligent tool curation.

Tools are frozen at Live session setup -- the client message oneof is exactly
``{setup, client_content, realtime_input, tool_response}``, with no tool-update
message -- so the only lever is choosing a good small set *before* connecting.

That choice matters more than it first appears, because declarations live in the
resident context and are therefore re-billed on every turn. Google's own guidance
is to "keep active set to 10-20 tools maximum" (the hard cap is 128), both for
cost and because selection accuracy degrades as the tool count grows.

Pipeline: compact -> measure -> rank -> pack under a token budget.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .catalog import Selection, ToolCandidate, matches_any

log = logging.getLogger(__name__)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def l2_normalize(vec: list[float]) -> list[float]:
    """Required after MRL truncation (output_dimensionality < native) before
    cosine similarity."""
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class Embedder:
    """Thin wrapper over ``client.models.embed_content`` with a memo cache."""

    def __init__(self, client: Any, model: str, dimensions: int | None = 768) -> None:
        self._client = client
        self._model = model
        self._dims = dimensions
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for text in texts:
            key = f"{task_type}:{self._dims}:{text}"
            if key in self._cache:
                out.append(self._cache[key])
                continue
            # gemini-embedding-001 on Vertex has historically accepted only one
            # instance per request, so we loop rather than batch.
            resp = self._client.models.embed_content(
                model=self._model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dims
                ),
            )
            vec = l2_normalize(list(resp.embeddings[0].values or []))
            self._cache[key] = vec
            out.append(vec)
        return out


class ToolCurator:
    def __init__(
        self,
        *,
        mode: str = "auto",
        max_tools: int = 20,
        budget_tokens: int = 6000,
        min_score: float = 0.25,
        dedupe_threshold: float = 0.92,
        pinned: list[str] | None = None,
        exclude: list[str] | None = None,
        allow: list[str] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.mode = mode
        self.max_tools = min(max_tools, 128)
        self.budget_tokens = budget_tokens
        self.min_score = min_score
        self.dedupe_threshold = dedupe_threshold
        self.pinned = pinned or []
        self.exclude = exclude or []
        self.allow = allow or []
        self._embedder = embedder

    # ------------------------------------------------------------------ select

    def select(self, catalog: list[ToolCandidate], purpose: str) -> Selection:
        catalog_tokens = sum(c.token_cost for c in catalog)
        catalog_size = len(catalog)
        dropped: list[ToolCandidate] = []

        # 1. Hard exclusions always win, whatever the score.
        pool: list[ToolCandidate] = []
        for cand in catalog:
            if matches_any(cand.exposed_name, self.exclude):
                cand.drop_reason = "excluded by tools.exclude"
                dropped.append(cand)
            else:
                pool.append(cand)

        if self.mode == "manual":
            return self._manual(pool, dropped, catalog_size, catalog_tokens)

        # 2. Pin flags.
        for cand in pool:
            cand.pinned = matches_any(cand.exposed_name, self.pinned)

        # 3. Semantic ranking.
        self._score(pool, purpose)

        # 4. Pack: pinned first (they reserve budget), then by value density.
        pinned = [c for c in pool if c.pinned]
        rest = sorted(
            (c for c in pool if not c.pinned),
            key=lambda c: (c.score / max(c.token_cost, 1), c.score),
            reverse=True,
        )

        selected: list[ToolCandidate] = []
        used = 0
        for cand in pinned:
            selected.append(cand)
            used += cand.token_cost

        for cand in rest:
            if len(selected) >= self.max_tools:
                cand.drop_reason = f"max_tools ({self.max_tools}) reached"
                dropped.append(cand)
                continue
            if cand.score < self.min_score:
                cand.drop_reason = f"score {cand.score:.2f} < min_score {self.min_score}"
                dropped.append(cand)
                continue
            dupe = self._near_duplicate(cand, selected)
            if dupe is not None:
                cand.drop_reason = f"near-duplicate of {dupe.exposed_name}"
                dropped.append(cand)
                continue
            if used + cand.token_cost > self.budget_tokens:
                cand.drop_reason = (
                    f"budget exhausted ({used}/{self.budget_tokens} tokens used)"
                )
                dropped.append(cand)
                continue
            selected.append(cand)
            used += cand.token_cost

        selected.sort(key=lambda c: (not c.pinned, -c.score, c.exposed_name))
        return Selection(
            selected=selected,
            dropped=dropped,
            total_tokens=used,
            catalog_size=catalog_size,
            catalog_tokens=catalog_tokens,
            mode="auto",
        )

    # ----------------------------------------------------------------- helpers

    def _manual(
        self,
        pool: list[ToolCandidate],
        dropped: list[ToolCandidate],
        catalog_size: int,
        catalog_tokens: int,
    ) -> Selection:
        selected: list[ToolCandidate] = []
        for cand in pool:
            if matches_any(cand.exposed_name, self.allow):
                selected.append(cand)
            else:
                cand.drop_reason = "not in tools.allow (manual mode)"
                dropped.append(cand)
        return Selection(
            selected=selected,
            dropped=dropped,
            total_tokens=sum(c.token_cost for c in selected),
            catalog_size=catalog_size,
            catalog_tokens=catalog_tokens,
            mode="manual",
        )

    def _score(self, pool: list[ToolCandidate], purpose: str) -> None:
        if not pool:
            return
        if self._embedder is None or not purpose.strip():
            # No ranking signal available: keep everything equally eligible and
            # let the budget decide. Cheapest-first is the sane tie-break.
            for cand in pool:
                cand.score = 1.0
            return
        try:
            docs = self._embedder.embed(
                [c.embedding_text for c in pool], task_type="RETRIEVAL_DOCUMENT"
            )
            query = self._embedder.embed([purpose], task_type="RETRIEVAL_QUERY")[0]
        except Exception as exc:
            log.warning("embedding unavailable (%s); falling back to budget-only curation", exc)
            for cand in pool:
                cand.score = 1.0
            return

        for cand, vec in zip(pool, docs, strict=False):
            cand.embedding = vec
            cand.score = cosine(query, vec)

    def _near_duplicate(
        self, cand: ToolCandidate, selected: list[ToolCandidate]
    ) -> ToolCandidate | None:
        if not cand.embedding:
            return None
        for other in selected:
            if other.embedding and cosine(cand.embedding, other.embedding) >= self.dedupe_threshold:
                return other
        return None
