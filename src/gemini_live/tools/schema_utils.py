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


def _resolve_ref(
    ref: Any, defs: dict[str, Any], seen: frozenset[str]
) -> dict[str, Any] | None:
    if not isinstance(ref, str):
        return None
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            if name in defs and name not in seen:
                target = defs[name]
                if isinstance(target, dict):
                    res = _sanitize(target, defs, seen | {name}, in_properties=False)
                    return res if isinstance(res, dict) else None
    return None


def _sanitize(
    schema: Any,
    defs: dict[str, Any],
    seen: frozenset[str],
    *,
    in_properties: bool = False,
) -> Any:
    if isinstance(schema, dict):
        if in_properties:
            # Keys here are tool parameter names (e.g. "title", "default"),
            # not JSON Schema keywords, so preserve every key name.
            return {
                k: _sanitize(v, defs, seen, in_properties=False)
                for k, v in schema.items()
            }

        merged: dict[str, Any] = {}
        if "$ref" in schema:
            resolved = _resolve_ref(schema["$ref"], defs, seen)
            if isinstance(resolved, dict):
                merged.update(resolved)

        for k, v in schema.items():
            if k in UNSUPPORTED_KEYS:
                continue
            if k == "properties" and isinstance(v, dict):
                merged[k] = _sanitize(v, defs, seen, in_properties=True)
            else:
                merged[k] = _sanitize(v, defs, seen, in_properties=False)

        if (
            "required" in merged
            and isinstance(merged["required"], list)
            and isinstance(merged.get("properties"), dict)
        ):
            props = merged["properties"]
            merged["required"] = [r for r in merged["required"] if r in props]

        return merged
    if isinstance(schema, list):
        return [_sanitize(v, defs, seen, in_properties=False) for v in schema]
    return schema


def sanitize_json_schema(schema: Any) -> Any:
    """Recursively inline $refs and drop keywords Gemini will not accept."""
    defs: dict[str, Any] = {}
    if isinstance(schema, dict):
        for key in ("$defs", "definitions"):
            sub = schema.get(key)
            if isinstance(sub, dict):
                defs.update(sub)
    return _sanitize(schema, defs, frozenset(), in_properties=False)


def truncate_description(text: str | None, max_chars: int) -> str:
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
