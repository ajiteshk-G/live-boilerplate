"""The core token-accounting contract.

``usageMetadata`` on the Live API is cumulative WITHIN a turn and resets at each
``turnComplete``. These tests exist mainly to lock in the consequence of that:
the obvious implementation is wrong, and wrong by a lot.
"""

from __future__ import annotations

from gemini_live.usage.accountant import TokenAccountant

from .conftest import FakeUsage, cumulative_turn, delta_turn


def test_naive_summing_overcounts_badly():
    """The bug this whole module exists to prevent.

    A turn that really cost 1,240 tokens looks like 3,000+ if you add up every
    message, because each message restates a running total.
    """
    snapshots = cumulative_turn(prompt=1000, response_steps=[40, 120, 240])

    naive = sum(s.total_token_count for s in snapshots)

    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)
    for s in snapshots:
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.total_tokens == 1240  # the last snapshot, not the sum
    assert naive == 3400
    # Not a rounding difference -- naive summing nearly triples the bill.
    assert naive > turn.total_tokens * 2


def test_prompt_tokens_are_never_summed_within_a_turn():
    """Prompt tokens are the resident context restated on every message.

    Summing them would report the context once per message received, which is
    the single most expensive way to get this wrong.
    """
    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)
    for s in cumulative_turn(prompt=5000, response_steps=[10, 20, 30, 40, 50]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.prompt_tokens == 5000  # not 25000


def test_monotonic_stream_resolves_to_last_wins():
    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)
    for s in cumulative_turn(prompt=100, response_steps=[10, 30, 90]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.accounting_mode == "last_wins"
    assert turn.response_tokens == 90


def test_non_monotonic_stream_resolves_to_delta_sum():
    """If the counter goes backwards, the server cannot be reporting a running
    total, so the values must be per-message increments and have to be added."""
    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)
    # 5 < 30 is the giveaway: a running total never shrinks mid-turn.
    for s in delta_turn(prompt=100, response_deltas=[10, 30, 5]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.accounting_mode == "delta_sum"
    assert turn.response_tokens == 45  # 10 + 30 + 5, not the final 5


def test_auto_detection_cannot_disambiguate_increasing_deltas():
    """A known, deliberate limitation -- documented rather than papered over.

    A delta stream whose increments happen to increase (10, 20, 30) is
    byte-identical to a cumulative stream (10, 20, 30). Nothing in-band can tell
    them apart, so `auto` assumes cumulative, which is the safer error: it
    under-reports rather than inventing tokens.

    This ambiguity is precisely why `glive calibrate-usage` exists -- it settles
    the question against a real model instead of guessing.
    """
    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)
    for s in delta_turn(prompt=100, response_deltas=[10, 20, 30]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.accounting_mode == "last_wins"
    assert turn.response_tokens == 30
    # The alternative reading is still recorded, so the data is not lost.
    assert turn.response_tokens_delta_sum == 60


def test_both_interpretations_are_always_recorded():
    """We do not yet know with certainty which interpretation is right for every
    model, so a turn always carries the delta-sum alongside the chosen figure."""
    acct = TokenAccountant(mode="last_wins", log_raw_snapshots=False)
    for s in cumulative_turn(prompt=100, response_steps=[10, 30, 90]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.response_tokens == 90  # chosen: final value
    assert turn.response_tokens_delta_sum == 130  # alternative: 10 + 30 + 90
    assert turn.snapshots == 3


def test_explicit_mode_overrides_detection():
    """A monotonic stream would auto-resolve to last_wins; forcing delta_sum
    must actually change the number, not just the label."""
    acct = TokenAccountant(mode="delta_sum", log_raw_snapshots=False)
    for s in cumulative_turn(prompt=100, response_steps=[10, 30, 90]):
        acct.observe(s)
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.accounting_mode == "delta_sum"
    assert turn.response_tokens == 130


def test_counters_reset_between_turns():
    """The reset at turnComplete is what makes summing committed turns correct."""
    acct = TokenAccountant(mode="auto", log_raw_snapshots=False)

    for s in cumulative_turn(prompt=1000, response_steps=[50]):
        acct.observe(s)
    first = acct.commit_turn()

    for s in cumulative_turn(prompt=1800, response_steps=[70]):
        acct.observe(s)
    second = acct.commit_turn()

    assert first is not None and second is not None
    assert first.turn_index == 0
    assert second.turn_index == 1
    assert second.snapshots == 1  # not 2: state was cleared

    session = acct.session_total()
    assert session.turns == 2
    assert session.prompt_tokens == 2800
    assert session.response_tokens == 120


def test_commit_with_no_snapshots_returns_none():
    """turnComplete can arrive with no usage attached; that must not create a
    phantom zero-token turn."""
    acct = TokenAccountant(log_raw_snapshots=False)
    assert acct.commit_turn() is None
    assert acct.session_total().turns == 0


def test_observe_ignores_none():
    """Most Live messages carry no usage_metadata at all."""
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(None)
    assert not acct.has_inflight
    assert acct.commit_turn() is None


def test_interrupted_turn_is_still_billed():
    """A barge-in ends a turn mid-generation. Those tokens were still charged,
    so committing on close must not lose them."""
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(prompt_token_count=900, response_token_count=25, total_token_count=925))

    assert acct.has_inflight
    turn = acct.commit_turn()
    assert turn is not None
    assert turn.total_tokens == 925


def test_turn_index_advances_only_on_commit():
    acct = TokenAccountant(log_raw_snapshots=False)
    assert acct.turn_index == 0
    acct.observe(FakeUsage(total_token_count=10))
    assert acct.turn_index == 0
    acct.commit_turn()
    assert acct.turn_index == 1
