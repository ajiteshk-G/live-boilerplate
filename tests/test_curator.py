"""Tool curation: ranking, budget packing, and the escape hatches."""

from __future__ import annotations

from gemini_live.tools.curator import ToolCurator, cosine, l2_normalize

from .conftest import StubEmbedder


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_is_safe_on_degenerate_input():
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0


def test_l2_normalize_produces_a_unit_vector():
    """Required after MRL truncation, or cosine similarity is meaningless."""
    vec = l2_normalize([3.0, 4.0])
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9


def test_l2_normalize_leaves_a_zero_vector_alone():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# ------------------------------------------------------------ budget packing


def test_selection_stops_at_max_tools(make_candidate):
    curator = ToolCurator(max_tools=3, budget_tokens=10**6)
    catalog = [make_candidate(f"s__t{i}", cost=10) for i in range(10)]

    selection = curator.select(catalog, purpose="")

    assert len(selection.selected) == 3
    assert len(selection.dropped) == 7
    assert all("max_tools" in c.drop_reason for c in selection.dropped)


def test_selection_stops_at_the_token_budget(make_candidate):
    curator = ToolCurator(max_tools=100, budget_tokens=250)
    catalog = [make_candidate(f"s__t{i}", cost=100) for i in range(10)]

    selection = curator.select(catalog, purpose="")

    assert selection.total_tokens <= 250
    assert len(selection.selected) == 2
    assert any("budget exhausted" in c.drop_reason for c in selection.dropped)


def test_saved_tokens_reports_what_curation_avoided(make_candidate):
    curator = ToolCurator(max_tools=2, budget_tokens=10**6)
    catalog = [make_candidate(f"s__t{i}", cost=500) for i in range(10)]

    selection = curator.select(catalog, purpose="")

    assert selection.catalog_tokens == 5000
    assert selection.total_tokens == 1000
    assert selection.saved_tokens == 4000


def test_projected_cost_multiplies_by_turns(make_candidate):
    """Declarations are re-billed every turn -- that is the number that matters."""
    curator = ToolCurator(max_tools=2, budget_tokens=10**6)
    selection = curator.select([make_candidate(f"s__t{i}", cost=500) for i in range(5)], purpose="")

    assert selection.projected_cost(turns=30) == 1000 * 30


# ------------------------------------------------------------------ excludes


def test_exclude_wins_over_everything(make_candidate):
    """Even a perfect score must not resurrect an excluded tool."""
    curator = ToolCurator(exclude=["*__delete_*"], max_tools=100)
    catalog = [
        make_candidate("fs__delete_file", cost=10),
        make_candidate("fs__read_file", cost=10),
    ]

    selection = curator.select(catalog, purpose="")

    assert [c.exposed_name for c in selection.selected] == ["fs__read_file"]
    assert selection.dropped[0].drop_reason == "excluded by tools.exclude"


def test_exclude_beats_pinned(make_candidate):
    curator = ToolCurator(exclude=["fs__danger"], pinned=["fs__danger"], max_tools=100)
    selection = curator.select([make_candidate("fs__danger")], purpose="")
    assert selection.selected == []


# -------------------------------------------------------------------- pinned


def test_pinned_tools_are_always_selected_even_over_budget(make_candidate):
    """A pinned tool is a deliberate instruction; silently dropping it would be
    worse than exceeding the advisory budget."""
    curator = ToolCurator(pinned=["s__critical"], max_tools=2, budget_tokens=10)
    catalog = [
        make_candidate("s__critical", cost=5000),
        make_candidate("s__other", cost=5),
    ]

    selection = curator.select(catalog, purpose="")

    assert "s__critical" in [c.exposed_name for c in selection.selected]


def test_pinned_tools_sort_first(make_candidate):
    curator = ToolCurator(pinned=["s__pinned"], max_tools=10, budget_tokens=10**6)
    catalog = [make_candidate("s__a"), make_candidate("s__pinned"), make_candidate("s__b")]

    selection = curator.select(catalog, purpose="")

    assert selection.selected[0].exposed_name == "s__pinned"


# ------------------------------------------------------------------ ranking


def test_relevant_tools_outrank_irrelevant_ones(make_candidate):
    """The core of 'intelligent': pick by meaning, not by config order."""
    embedder = StubEmbedder(
        {
            "search the documentation": [1.0, 0.0, 0.0],  # the purpose
            "docs__search": [1.0, 0.0, 0.0],  # on point
            "fs__chmod": [0.0, 1.0, 0.0],  # unrelated
        }
    )
    curator = ToolCurator(max_tools=1, budget_tokens=10**6, embedder=embedder, min_score=0.0)
    catalog = [make_candidate("fs__chmod", cost=10), make_candidate("docs__search", cost=10)]

    selection = curator.select(catalog, purpose="search the documentation")

    assert [c.exposed_name for c in selection.selected] == ["docs__search"]


def test_tools_below_min_score_are_dropped(make_candidate):
    embedder = StubEmbedder(
        {"find docs": [1.0, 0.0, 0.0], "docs__search": [1.0, 0.0, 0.0]}
    )
    curator = ToolCurator(max_tools=10, budget_tokens=10**6, embedder=embedder, min_score=0.5)
    catalog = [make_candidate("docs__search"), make_candidate("fs__chmod")]

    selection = curator.select(catalog, purpose="find docs")

    assert [c.exposed_name for c in selection.selected] == ["docs__search"]
    assert "min_score" in selection.dropped[0].drop_reason


def test_near_duplicate_tools_are_dropped(make_candidate):
    """Two MCP servers exposing the same capability should not both be paid for."""
    same = [1.0, 0.0, 0.0]
    embedder = StubEmbedder(
        {"search": same, "a__search": same, "b__search": same}
    )
    curator = ToolCurator(
        max_tools=10, budget_tokens=10**6, embedder=embedder, min_score=0.0,
        dedupe_threshold=0.92,
    )
    catalog = [make_candidate("a__search"), make_candidate("b__search")]

    selection = curator.select(catalog, purpose="search")

    assert len(selection.selected) == 1
    assert "near-duplicate" in selection.dropped[0].drop_reason


def test_embedding_failure_degrades_to_budget_only_curation(make_candidate):
    """An embedding outage must not take the voice agent down with it."""
    curator = ToolCurator(
        max_tools=2, budget_tokens=10**6,
        embedder=StubEmbedder({}, fail=True), min_score=0.0,
    )
    catalog = [make_candidate(f"s__t{i}") for i in range(5)]

    selection = curator.select(catalog, purpose="anything")

    assert len(selection.selected) == 2
    assert all(c.score == 1.0 for c in selection.selected)


def test_no_embedder_still_produces_a_selection(make_candidate):
    curator = ToolCurator(max_tools=2, budget_tokens=10**6, min_score=0.0)
    selection = curator.select([make_candidate(f"s__t{i}") for i in range(5)], purpose="x")
    assert len(selection.selected) == 2


def test_value_density_prefers_cheap_tools_at_equal_relevance(make_candidate):
    """With nothing to separate them on meaning, fit more tools in the budget."""
    curator = ToolCurator(max_tools=1, budget_tokens=10**6, min_score=0.0)
    catalog = [make_candidate("s__expensive", cost=1000), make_candidate("s__cheap", cost=10)]

    selection = curator.select(catalog, purpose="")

    assert selection.selected[0].exposed_name == "s__cheap"


# ------------------------------------------------------------- manual mode


def test_manual_mode_uses_only_the_allow_list(make_candidate):
    curator = ToolCurator(mode="manual", allow=["docs__*"], max_tools=100)
    catalog = [
        make_candidate("docs__search"),
        make_candidate("docs__fetch"),
        make_candidate("fs__rm"),
    ]

    selection = curator.select(catalog, purpose="ignored in manual mode")

    assert selection.mode == "manual"
    assert sorted(c.exposed_name for c in selection.selected) == ["docs__fetch", "docs__search"]
    assert selection.dropped[0].drop_reason == "not in tools.allow (manual mode)"


def test_manual_mode_still_honours_exclude(make_candidate):
    curator = ToolCurator(mode="manual", allow=["fs__*"], exclude=["fs__rm"], max_tools=100)
    selection = curator.select(
        [make_candidate("fs__ls"), make_candidate("fs__rm")], purpose=""
    )
    assert [c.exposed_name for c in selection.selected] == ["fs__ls"]


# ----------------------------------------------------------------- edge cases


def test_empty_catalog_is_handled(make_candidate):
    selection = ToolCurator().select([], purpose="anything")
    assert selection.selected == []
    assert selection.catalog_size == 0
    assert selection.projected_cost(10) == 0


def test_max_tools_is_clamped_to_the_api_limit():
    """The API rejects more than 128 declarations outright."""
    assert ToolCurator(max_tools=500).max_tools == 128
