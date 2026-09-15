"""Schema sanitisation and description compaction.

Gemini rejects several standard JSON-Schema keywords that MCP servers emit
freely. Getting this wrong produces an opaque 400 at connect time.
"""

from __future__ import annotations

from gemini_live.tools.schema_utils import (
    UNSUPPORTED_KEYS,
    normalize_schema,
    sanitize_json_schema,
    truncate_description,
)


def test_unsupported_keywords_are_stripped():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SearchArgs",
        "type": "object",
        "additionalProperties": False,
        "properties": {"q": {"type": "string", "default": "", "title": "Query"}},
    }
    cleaned = sanitize_json_schema(schema)

    assert "$schema" not in cleaned
    assert "title" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "default" not in cleaned["properties"]["q"]
    # The parts Gemini actually needs survive.
    assert cleaned["type"] == "object"
    assert cleaned["properties"]["q"]["type"] == "string"


def test_stripping_recurses_into_nested_structures():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "title": "Item", "properties": {}},
            }
        },
    }
    cleaned = sanitize_json_schema(schema)
    assert "title" not in cleaned["properties"]["items"]["items"]


def test_stripping_recurses_through_lists():
    cleaned = sanitize_json_schema({"anyOf": [{"type": "string", "title": "x"}]})
    assert "title" not in cleaned["anyOf"][0]


def test_every_listed_keyword_is_actually_removed():
    schema = {k: "x" for k in UNSUPPORTED_KEYS} | {"type": "object"}
    assert sanitize_json_schema(schema) == {"type": "object"}


def test_required_and_enum_are_preserved():
    """These are load-bearing for correct tool calls and must not be stripped."""
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
        "required": ["mode"],
    }
    cleaned = sanitize_json_schema(schema)
    assert cleaned["required"] == ["mode"]
    assert cleaned["properties"]["mode"]["enum"] == ["a", "b"]


# ------------------------------------------------------------- normalization


def test_empty_schema_becomes_a_valid_object_schema():
    """A no-argument MCP tool often reports {} or None; Gemini needs a shape."""
    expected = {"type": "object", "properties": {}}
    assert normalize_schema({}) == expected
    assert normalize_schema(None) == expected
    assert normalize_schema("nonsense") == expected


def test_schema_emptied_by_sanitising_still_becomes_valid():
    assert normalize_schema({"$schema": "x", "title": "y"}) == {"type": "object", "properties": {}}


def test_object_schema_gains_missing_properties_key():
    assert normalize_schema({"type": "object"}) == {"type": "object", "properties": {}}


def test_non_object_schema_is_left_alone():
    assert normalize_schema({"type": "string"}) == {"type": "string"}


# ------------------------------------------------------------- descriptions


def test_short_description_is_untouched():
    assert truncate_description("Search the docs.", 200) == "Search the docs."


def test_whitespace_is_collapsed():
    """MCP descriptions are often indented multi-line strings; the newlines are
    billed tokens that carry no meaning."""
    assert truncate_description("Search\n   the\t\tdocs.", 200) == "Search the docs."


def test_truncation_prefers_a_sentence_boundary():
    text = "Searches the documentation corpus for a query. Supports boolean operators too."
    result = truncate_description(text, 60)
    assert result == "Searches the documentation corpus for a query."


def test_sentence_boundary_is_ignored_when_it_would_waste_the_budget():
    """Cutting at the first full stop is only worth it if a useful amount of
    text survives. A two-word opening sentence would throw away most of the
    allowance and leave a description that says nothing, so a word boundary is
    used instead.
    """
    text = "Searches. Finds matching documents in the indexed corpus using ranking."
    result = truncate_description(text, 50)
    assert result != "Searches."
    assert result.endswith("...")


def test_truncation_falls_back_to_a_word_boundary():
    text = "Searches the corpus using an advanced ranking algorithm with no sentence breaks"
    result = truncate_description(text, 30)
    assert len(result) <= 33
    assert result.endswith("...")
    assert not result.replace("...", "").endswith(" ")


def test_truncation_handles_a_single_long_word():
    result = truncate_description("x" * 500, 20)
    assert len(result) == 20


def test_empty_description_is_safe():
    assert truncate_description("", 100) == ""
    assert truncate_description(None, 100) == ""


def test_zero_max_chars_disables_truncation():
    assert truncate_description("anything at all", 0) == "anything at all"


def test_compaction_measurably_reduces_declaration_size(make_candidate):
    """Descriptions are re-billed every turn, so this saving recurs."""
    verbose = (
        "Searches the documentation corpus. " + "This is a long worked example. " * 20
    )
    cand = make_candidate("docs__search", description=verbose)
    before = len(cand.description)

    cand.compacted(max_desc_chars=200)

    assert len(cand.description) < before / 2


def test_compaction_also_normalizes_the_schema(make_candidate):
    cand = make_candidate("t", schema={"$schema": "x", "type": "object", "title": "T"})
    cand.compacted(max_desc_chars=200)
    assert "$schema" not in cand.input_schema
    assert "title" not in cand.input_schema


def test_embedding_text_includes_name_description_and_parameters(make_candidate):
    """Parameter names are a strong relevance signal and cost nothing to include."""
    cand = make_candidate(
        "docs__search",
        description="Find docs",
        schema={"type": "object", "properties": {"query": {}, "limit": {}}},
    )
    text = cand.embedding_text

    assert "docs__search" in text
    assert "Find docs" in text
    assert "limit, query" in text


def test_embedding_text_without_parameters(make_candidate):
    cand = make_candidate("t__now", description="Current time")
    assert cand.embedding_text == "t__now: Current time"


def test_properties_named_title_or_default_are_preserved():
    """A tool parameter named 'title' or 'default' inside properties must not be deleted."""
    schema = {
        "type": "object",
        "title": "CreateDocArgs",
        "properties": {
            "title": {"type": "string", "title": "Doc Title", "default": "Untitled"},
            "default": {"type": "boolean", "default": False},
        },
        "required": ["title", "default"],
    }
    cleaned = sanitize_json_schema(schema)

    assert "title" not in cleaned  # top-level schema title stripped
    assert "title" in cleaned["properties"]
    assert "default" in cleaned["properties"]
    assert cleaned["properties"]["title"] == {"type": "string"}
    assert cleaned["properties"]["default"] == {"type": "boolean"}
    assert cleaned["required"] == ["title", "default"]


def test_refs_and_defs_are_inlined_before_defs_stripped():
    """Pydantic/FastMCP schemas use $defs + $ref; stripping without inlining leaves empty {}."""
    schema = {
        "type": "object",
        "$defs": {
            "FilterSpec": {
                "type": "object",
                "title": "FilterSpec",
                "properties": {"field": {"type": "string"}, "op": {"type": "string"}},
                "required": ["field"],
            }
        },
        "properties": {
            "filter": {"$ref": "#/$defs/FilterSpec", "description": "Filter criteria"}
        },
    }
    cleaned = sanitize_json_schema(schema)

    assert "$defs" not in cleaned
    assert "$ref" not in cleaned["properties"]["filter"]
    assert cleaned["properties"]["filter"]["type"] == "object"
    assert cleaned["properties"]["filter"]["description"] == "Filter criteria"
    assert "field" in cleaned["properties"]["filter"]["properties"]
    assert cleaned["properties"]["filter"]["required"] == ["field"]

