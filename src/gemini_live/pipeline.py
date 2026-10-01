"""Assemble the tool pipeline: ingest -> compact -> measure -> select."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from .settings.schema import AppConfig
from .tools.builtins import build_builtin_candidates
from .tools.catalog import Selection, ToolCandidate
from .tools.cost import DeclarationCostMeter
from .tools.curator import Embedder, ToolCurator
from .tools.mcp_toolset import McpToolsetManager
from .tools.registry import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class ToolPipelineResult:
    registry: ToolRegistry
    selection: Selection
    catalog: list[ToolCandidate]
    mcp_failures: dict[str, str]


class ToolPipeline:
    """Async context manager owning MCP connections for the pipeline's lifetime."""

    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self._cfg = cfg
        self._client = client
        self._stack = AsyncExitStack()
        self._mcp: McpToolsetManager | None = None

    async def __aenter__(self) -> ToolPipeline:
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def build(self) -> ToolPipelineResult:
        cfg = self._cfg
        curation = cfg.tools.curation

        # 1. Ingest -----------------------------------------------------------
        catalog: list[ToolCandidate] = build_builtin_candidates(cfg.tools)
        failures: dict[str, str] = {}
        if cfg.tools.mcp:
            self._mcp = await self._stack.enter_async_context(
                McpToolsetManager(cfg.tools.mcp)
            )
            catalog.extend(self._mcp.candidates)
            failures = dict(self._mcp.failures)

        # 2. Compact ----------------------------------------------------------
        for cand in catalog:
            cand.compacted(curation.max_desc_chars)

        # 3. Measure ----------------------------------------------------------
        meter = DeclarationCostMeter(
            self._client,
            curation.cost_model,
            enabled=self._client is not None,
        )

        def _measure_all() -> None:
            for cand in catalog:
                cand.token_cost = meter.measure(
                    cand.exposed_name, cand.description, cand.input_schema
                )
            meter.flush()

        await asyncio.to_thread(_measure_all)

        # 4. Select -----------------------------------------------------------
        embedder = (
            Embedder(self._client, curation.embedding_model, curation.embedding_dimensions)
            if self._client is not None and curation.mode == "auto"
            else None
        )
        curator = ToolCurator(
            mode=curation.mode,
            max_tools=curation.max_tools,
            budget_tokens=curation.budget_tokens,
            min_score=curation.min_score,
            dedupe_threshold=curation.dedupe_threshold,
            pinned=cfg.tools.pinned,
            exclude=cfg.tools.exclude,
            allow=cfg.tools.allow,
            embedder=embedder,
        )
        selection = await asyncio.to_thread(curator.select, catalog, cfg.curation_purpose())

        log.info(
            "tools: %d/%d selected, %d tokens per turn (catalog would cost %d)",
            len(selection.selected),
            selection.catalog_size,
            selection.total_tokens,
            selection.catalog_tokens,
        )
        return ToolPipelineResult(
            registry=ToolRegistry.from_selection(
                selection, scheduling=cfg.tools.scheduling
            ),
            selection=selection,
            catalog=catalog,
            mcp_failures=failures,
        )
