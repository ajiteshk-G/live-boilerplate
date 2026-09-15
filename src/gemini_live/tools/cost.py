"""Measure the exact token cost of function declarations.

Why this exists: tool declarations sit in the resident context and are re-billed
on EVERY turn. Guessing their size is not good enough -- we measure.

``CountTokensConfig.tools`` is supported on the Vertex path
(``_CountTokensConfig_to_vertex`` maps ``['tools']``); the Gemini Developer API
converter raises, which is irrelevant here because this boilerplate is
Vertex-only. Live sessions do not support ``countTokens`` at all, so measurement
happens against an ordinary model and is cached on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROBE = "x"


def _digest(name: str, description: str, schema: Any, model: str = "") -> str:
    blob = json.dumps(
        {"m": model, "n": name, "d": description, "s": schema},
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


class DeclarationCostMeter:
    """Per-declaration token cost, measured once and cached."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        cache_path: str | Path | None = ".glive_cache/tool_costs.json",
        enabled: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._enabled = enabled
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, int] = {}
        self._base: int | None = None
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except (OSError, ValueError):
                self._cache = {}

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        except OSError as exc:
            log.debug("could not persist tool cost cache: %s", exc)

    def _count(self, tools: list[Any] | None) -> int:
        from google.genai import types

        config = types.CountTokensConfig(tools=tools) if tools else None
        resp = self._client.models.count_tokens(
            model=self._model, contents=_PROBE, config=config
        )
        return int(getattr(resp, "total_tokens", 0) or 0)

    def _baseline(self) -> int:
        if self._base is None:
            self._base = self._count(None)
        return self._base

    def measure(self, name: str, description: str, schema: dict[str, Any]) -> int:
        """Token cost of a single declaration. Returns a heuristic if unavailable."""
        key = _digest(name, description, schema, self._model)
        if key in self._cache:
            return self._cache[key]

        estimate = self._estimate(name, description, schema)
        if not self._enabled:
            # Never persist offline heuristics to disk, or they will permanently
            # shadow real token counts on subsequent online runs.
            return estimate

        try:
            from google.genai import types

            decl = types.FunctionDeclaration(
                name=name,
                description=description,
                parameters_json_schema=schema,
            )
            one_tool = [types.Tool(function_declarations=[decl])]
            measured = self._count(one_tool) - self._baseline()
            if measured > 0:
                self._cache[key] = measured
                self._save_cache()
                return measured
        except Exception as exc:
            # Never let measurement break startup; fall back to the estimate.
            log.debug("count_tokens unavailable for %s (%s); using estimate", name, exc)
            self._enabled = False

        return estimate

    @staticmethod
    def _estimate(name: str, description: str, schema: dict[str, Any]) -> int:
        """Offline fallback: ~4 characters per token over the serialised declaration."""
        blob = name + description + json.dumps(schema, default=str)
        return max(1, len(blob) // 4)

    def flush(self) -> None:
        self._save_cache()
