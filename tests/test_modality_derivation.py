"""Per-modality reporting.

The API exposes exactly FOUR ModalityTokenCount arrays: prompt, cache, response
and tool-use. There is no `thoughts_tokens_details` and no
`total_tokens_details`. These tests exist to stop someone "helpfully" inventing
them, and to keep the derived total honestly labelled.
"""

from __future__ import annotations

from gemini_live.usage.accountant import TokenAccountant
from gemini_live.usage.models import UNKNOWN_MODALITY, TurnUsage, merge_modalities

from .conftest import FakeModality, FakeUsage


def test_total_by_modality_is_derived_from_the_four_real_arrays():
    turn = TurnUsage(
        turn_index=0,
        prompt_by_modality={"AUDIO": 100, "TEXT": 50},
        cache_by_modality={"TEXT": 10},
        response_by_modality={"AUDIO": 200},
        tool_use_by_modality={"TEXT": 5},
    )
    assert turn.total_by_modality == {"AUDIO": 300, "TEXT": 65}


def test_thoughts_have_no_modality_breakdown():
    """Thinking tokens are a scalar. If a breakdown ever appears in the API this
    test should fail and force a deliberate update, rather than the number being
    silently fabricated."""
    turn = TurnUsage(turn_index=0, thoughts_tokens=500)
    assert not hasattr(turn, "thoughts_by_modality")
    assert turn.thoughts_tokens == 500
    # Thinking is deliberately absent from the derived total-by-modality.
    assert turn.total_by_modality == {}


def test_derived_total_is_labelled_as_derived_in_the_event():
    """UI and logs must not present a derived figure as server-reported."""
    turn = TurnUsage(turn_index=0, response_by_modality={"AUDIO": 10})
    event = turn.as_event()
    assert "total_derived" in event["by_modality"]
    assert "total" not in event["by_modality"]


def test_none_modality_is_bucketed_not_dropped():
    """Real recorded fixtures contain entries with modality=None. Dropping them
    would under-report; crashing would be worse."""
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(
        FakeUsage(
            response_token_count=30,
            response_tokens_details=[
                FakeModality("AUDIO", 20),
                FakeModality(None, 10),
            ],
        )
    )
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.response_by_modality == {"AUDIO": 20, UNKNOWN_MODALITY: 10}
    # Nothing vanished.
    assert sum(turn.response_by_modality.values()) == 30


def test_repeated_modalities_are_summed():
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(
        FakeUsage(
            prompt_tokens_details=[
                FakeModality("TEXT", 5),
                FakeModality("TEXT", 7),
            ]
        )
    )
    turn = acct.commit_turn()
    assert turn is not None
    assert turn.prompt_by_modality == {"TEXT": 12}


def test_enum_like_modalities_are_unwrapped():
    """The SDK hands back enum members, not plain strings."""

    class FakeEnum:
        value = "AUDIO"

    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(response_tokens_details=[FakeModality(FakeEnum(), 42)]))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.response_by_modality == {"AUDIO": 42}


def test_missing_details_arrays_are_empty_not_an_error():
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(prompt_token_count=10, total_token_count=10))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.prompt_by_modality == {}
    assert turn.total_by_modality == {}


def test_session_modality_totals_accumulate_across_turns():
    acct = TokenAccountant(log_raw_snapshots=False)
    for _ in range(3):
        acct.observe(FakeUsage(response_tokens_details=[FakeModality("AUDIO", 100)]))
        acct.commit_turn()

    assert acct.session_total().by_modality == {"AUDIO": 300}


def test_merge_modalities_is_additive():
    assert merge_modalities({"A": 1}, {"A": 2, "B": 3}, {}) == {"A": 3, "B": 3}
