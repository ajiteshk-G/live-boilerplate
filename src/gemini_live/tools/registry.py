"""Tool registry: declarations for the model, and dispatch for tool calls.

The Live API never executes tools for you. Unlike ``generate_content``, the
session only emits a ``toolCall`` message and waits for you to reply with
``send_tool_response``. This registry owns both halves of that contract.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .catalog import Selection, ToolCandidate

log = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_S = 30.0


class ToolRegistry:
    """Maps exposed tool names to declarations and invocation handlers."""

    def __init__(
        self,
        candidates: list[ToolCandidate],
        *,
        timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    ) -> None:
        self._by_name: dict[str, ToolCandidate] = {c.exposed_name: c for c in candidates}
        self._timeout = timeout_s

    @classmethod
    def from_selection(cls, selection: Selection, **kwargs: Any) -> ToolRegistry:
        return cls(selection.selected, **kwargs)

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> list[str]:
        return sorted(self._by_name)

    @property
    def total_declaration_tokens(self) -> int:
        return sum(c.token_cost for c in self._by_name.values())

    def declarations(self) -> list[Any]:
        """One ``types.Tool`` bundling every permitted declaration (or [])."""
        from google.genai import types

        if not self._by_name:
            return []
        decls = [
            types.FunctionDeclaration(
                name=c.exposed_name,
                description=c.description,
                parameters_json_schema=c.input_schema,
            )
            for c in self._by_name.values()
        ]
        return [types.Tool(function_declarations=decls)]

    async def dispatch(self, function_call: Any) -> Any:
        """Execute one function call and build its ``FunctionResponse``.

        Never raises: a tool failure is reported back to the model as a
        structured error so the conversation can continue gracefully.
        """
        from google.genai import types

        name = getattr(function_call, "name", "") or ""
        call_id = getattr(function_call, "id", None)
        args = dict(getattr(function_call, "args", None) or {})

        tool = self._by_name.get(name)
        if tool is None:
            # Defence in depth: the model occasionally invents a name, and a
            # curated session deliberately excludes most of the catalog.
            log.warning("model called unavailable tool %r", name)
            payload: dict[str, Any] = {
                "error": f"Tool '{name}' is not available in this session."
            }
        else:
            try:
                payload = await asyncio.wait_for(tool.invoke(args), timeout=self._timeout)
            except TimeoutError:
                payload = {"error": f"Tool '{name}' timed out after {self._timeout:.0f}s."}
            except Exception as exc:
                log.exception("tool %r raised", name)
                payload = {"error": f"{type(exc).__name__}: {exc}"}

        # Echoing the call id is mandatory: the Live API matches responses to
        # calls by id, and omitting it wedges the turn.
        return types.FunctionResponse(id=call_id, name=name, response=payload)

    async def dispatch_all(self, function_calls: list[Any]) -> list[Any]:
        if not function_calls:
            return []
        return list(
            await asyncio.gather(*(self.dispatch(fc) for fc in function_calls))
        )
