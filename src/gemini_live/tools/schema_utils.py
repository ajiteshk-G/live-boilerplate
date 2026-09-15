"""JSON-Schema sanitisation and declaration building.

Gemini rejects several standard JSON-Schema keywords that MCP servers emit
freely, so every schema must be filtered before it reaches a
``FunctionDeclaration``. This mirrors what the SDK's private
``_mcp_utils._filter_to_supported_schema`` does, but as our own code so we do not
depend on a private module that can change between releases.
"""

from __future__ import annotations

import re
from typing import Any

UNSUPPORTED_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "additionalProperties",
        "default",
        "title",
        "examples",
        "const",
        "patternProperties",
        "unevaluatedProperties",
    }
)

_WS = re.compile(r"\s+")


def sanitize_json_schema(schema: Any) -> Any:
    """Recursively drop keywords Gemini will not accept."""
    if isinstance(schema, dict):
        return {
            k: sanitize_json_schema(v) for k, v in schema.items() if k not in UNSUPPORTED_KEYS
        }
    if isinstance(schema, list):
        return [sanitize_json_schema(v) for v in schema]
    return schema


def truncate_description(text: str, max_chars: int) -> str:
    """Shorten a description, preferring to cut at a sentence boundary.

    MCP servers routinely ship 500+ character descriptions with worked examples.
    Trimming typically halves declaration cost with no measurable loss of
    tool-selection quality -- and because declarations are re-billed every turn,
    that saving recurs for the whole session.
    """
    text = _WS.sub(" ", (text or "").strip())
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    window = text[: max_chars + 1]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= max_chars // 2:
        return window[: cut + 1].strip()

    cut = window.rfind(" ")
    if cut <= 0:
        return text[:max_chars].strip()
    return window[:cut].strip() + "..."


def empty_object_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


def normalize_schema(schema: Any) -> dict[str, Any]:
    """Sanitise and guarantee a usable object schema."""
    if not isinstance(schema, dict) or not schema:
        return empty_object_schema()
    cleaned = sanitize_json_schema(schema)
    if not isinstance(cleaned, dict) or not cleaned:
        return empty_object_schema()
    cleaned.setdefault("type", "object")
    if cleaned.get("type") == "object":
        cleaned.setdefault("properties", {})
    return cleaned
