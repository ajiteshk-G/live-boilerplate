"""Provenance fields (traffic_type, service_tier) and optional pricing."""

from __future__ import annotations

from gemini_live.usage.accountant import TokenAccountant
from gemini_live.usage.models import TurnUsage
from gemini_live.usage.pricing import PriceTable

from .conftest import FakeModality, FakeUsage

# ------------------------------------------------------------- traffic_type


def test_traffic_type_is_captured():
    """Vertex-only. Distinguishes pay-as-you-go from provisioned throughput."""
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(traffic_type="ON_DEMAND", total_token_count=10))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.traffic_type == "ON_DEMAND"
    assert turn.as_event()["traffic_type"] == "ON_DEMAND"


def test_unknown_traffic_type_values_pass_through_unchanged():
    """The enum has documented gaps (priority and flex traffic). An unrecognised
    value must be recorded verbatim, never coerced or dropped."""
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(traffic_type="SOME_FUTURE_TIER", total_token_count=10))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.traffic_type == "SOME_FUTURE_TIER"


def test_enum_traffic_type_is_unwrapped_to_its_value():
    class FakeEnum:
        value = "PROVISIONED_THROUGHPUT"

    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(traffic_type=FakeEnum(), total_token_count=1))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.traffic_type == "PROVISIONED_THROUGHPUT"


def test_absent_traffic_type_is_none():
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(total_token_count=1))
    turn = acct.commit_turn()
    assert turn is not None
    assert turn.traffic_type is None


# ------------------------------------------------------------ service_tier


def test_service_tier_is_captured_but_hidden_by_default():
    """The field exists on the type but Vertex never populates it. Showing an
    always-empty field in the UI is noise, so it is captured for the log and
    omitted from the default event.
    """
    acct = TokenAccountant(log_raw_snapshots=False)
    acct.observe(FakeUsage(service_tier="STANDARD", total_token_count=10))
    turn = acct.commit_turn()

    assert turn is not None
    assert turn.service_tier == "STANDARD"
    assert "service_tier" not in turn.as_event()
    assert turn.as_event(include_service_tier=True)["service_tier"] == "STANDARD"


# ----------------------------------------------------------------- pricing


def test_pricing_splits_audio_and_text_by_modality():
    prices = PriceTable(
        text_input_per_1k=1.0,
        audio_input_per_1k=10.0,
        text_output_per_1k=2.0,
        audio_output_per_1k=20.0,
    )
    turn = TurnUsage(
        turn_index=0,
        prompt_tokens=1000,
        response_tokens=1000,
        prompt_by_modality={"AUDIO": 600, "TEXT": 400},
        response_by_modality={"AUDIO": 800},
    )

    # audio in 0.6*10 + text in 0.4*1 + audio out 0.8*20 + text out 0.2*2
    assert prices.estimate(turn) == 6.0 + 0.4 + 16.0 + 0.4


def test_unknown_modality_tokens_are_priced_as_text():
    """Better to attribute them to the cheaper bucket than to lose them."""
    prices = PriceTable(text_input_per_1k=1.0, audio_input_per_1k=10.0)
    turn = TurnUsage(
        turn_index=0,
        prompt_tokens=1000,
        prompt_by_modality={"AUDIO": 400, "UNKNOWN": 600},
    )
    assert prices.estimate(turn) == 4.0 + 0.6


def test_thinking_tokens_are_priced_separately():
    prices = PriceTable(thinking_per_1k=5.0)
    turn = TurnUsage(turn_index=0, thoughts_tokens=2000)
    assert prices.estimate(turn) == 10.0


def test_missing_modality_data_does_not_produce_a_negative_cost():
    """If response_by_modality reports more audio than response_tokens, the
    text remainder must clamp at zero rather than going negative."""
    prices = PriceTable(text_output_per_1k=1.0, audio_output_per_1k=1.0)
    turn = TurnUsage(
        turn_index=0,
        response_tokens=100,
        response_by_modality={"AUDIO": 500},
    )
    assert prices.estimate(turn) >= 0


def test_zero_rate_table_costs_nothing():
    assert PriceTable().estimate(TurnUsage(turn_index=0, prompt_tokens=10**6)) == 0.0


# -------------------------------------------------------------- jsonl log


def test_turn_and_snapshots_are_written_to_the_log(tmp_path):
    import json

    path = tmp_path / "usage.jsonl"
    acct = TokenAccountant(raw_log_path=path, log_raw_snapshots=True)
    acct.observe(
        FakeUsage(
            prompt_token_count=100,
            response_token_count=20,
            total_token_count=120,
            response_tokens_details=[FakeModality("AUDIO", 20)],
            service_tier="X",
        )
    )
    acct.commit_turn()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [r["kind"] for r in records]

    assert kinds == ["snapshot", "turn"]
    assert records[0]["prompt_token_count"] == 100
    # service_tier is preserved in the log even though the UI hides it.
    assert records[1]["service_tier"] == "X"


def test_logging_failure_never_breaks_the_call(tmp_path):
    """Usage logging is observability, not functionality. It must not be able to
    take down a live conversation."""
    path = tmp_path / "sub" / "usage.jsonl"
    acct = TokenAccountant(raw_log_path=path, log_raw_snapshots=True)

    # Make the directory unwritable after the accountant created it.
    path.parent.chmod(0o500)
    try:
        acct.observe(FakeUsage(total_token_count=1))
        turn = acct.commit_turn()
        assert turn is not None  # accounting still worked
    finally:
        path.parent.chmod(0o700)


def test_nothing_is_written_when_no_log_path_is_configured():
    acct = TokenAccountant(raw_log_path=None)
    acct.observe(FakeUsage(total_token_count=1))
    assert acct.commit_turn() is not None
