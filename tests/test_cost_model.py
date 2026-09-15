"""Cost model: context rent, compression detection, and alerts."""

from __future__ import annotations

from gemini_live.usage.cost_model import CostModel
from gemini_live.usage.models import TurnUsage


def turn(index: int, prompt: int, response: int = 100, **kw) -> TurnUsage:
    return TurnUsage(
        turn_index=index,
        prompt_tokens=prompt,
        response_tokens=response,
        total_tokens=prompt + response + kw.get("thoughts_tokens", 0),
        **kw,
    )


# ------------------------------------------------------------- context rent


def test_context_rent_is_the_prompt_token_count():
    """Every turn re-bills the whole resident context; that is the prompt count."""
    t = turn(0, prompt=5000)
    assert t.context_rent == 5000


def test_marginal_tokens_exclude_re_read_context():
    t = TurnUsage(
        turn_index=0,
        prompt_tokens=5000,
        response_tokens=200,
        thoughts_tokens=50,
        tool_use_prompt_tokens=30,
        total_tokens=5280,
    )
    assert t.marginal_tokens == 280


def test_rent_ratio_exposes_a_conversation_paying_mostly_for_history():
    cheap = TurnUsage(turn_index=0, prompt_tokens=100, response_tokens=900, total_tokens=1000)
    expensive = TurnUsage(turn_index=9, prompt_tokens=9500, response_tokens=500, total_tokens=10000)

    assert cheap.rent_ratio == 0.10
    assert expensive.rent_ratio == 0.95


def test_rent_ratio_is_zero_when_nothing_was_billed():
    assert TurnUsage(turn_index=0).rent_ratio == 0.0


# ------------------------------------------------------- compression events


def test_growth_delta_is_zero_on_the_first_turn():
    model = CostModel()
    growth = model.observe_turn(turn(0, prompt=1000))
    assert growth.delta == 0
    assert growth.compression_event is False


def test_normal_growth_is_not_reported_as_compression():
    model = CostModel()
    model.observe_turn(turn(0, prompt=1000))
    growth = model.observe_turn(turn(1, prompt=1800))
    assert growth.delta == 800
    assert growth.compression_event is False


def test_large_drop_in_prompt_tokens_is_inferred_as_compression():
    """The API never announces compression, so a sizeable drop in context is the
    only available fingerprint."""
    model = CostModel(compression_epsilon=256)
    model.observe_turn(turn(0, prompt=26000))
    growth = model.observe_turn(turn(1, prompt=13000))

    assert growth.compression_event is True
    assert growth.delta == -13000


def test_small_dip_is_not_mistaken_for_compression():
    """Prompt counts wobble slightly turn to turn; epsilon stops false alarms."""
    model = CostModel(compression_epsilon=256)
    model.observe_turn(turn(0, prompt=10000))
    growth = model.observe_turn(turn(1, prompt=9900))

    assert growth.compression_event is False


# ----------------------------------------------------- tool cost projection


def test_tool_declarations_are_projected_as_a_recurring_charge():
    """The headline insight: a 6k tool payload is not a 6k cost. It is 6k
    *per turn*, for the whole session."""
    model = CostModel(tool_declaration_tokens=6000)

    first = model.observe_turn(turn(0, prompt=7000))
    assert first.projected_tool_cost == 6000

    for i in range(1, 30):
        growth = model.observe_turn(turn(i, prompt=7000 + i * 100))

    assert growth.projected_tool_cost == 6000 * 30  # 180,000 billed tokens


def test_cost_drivers_separate_per_turn_from_cumulative():
    model = CostModel(tool_declaration_tokens=4000, system_instruction_tokens=500)
    drivers = model.cost_drivers(turns=10)

    assert drivers["tool_declarations_per_turn"] == 4000
    assert drivers["tool_declarations_cumulative"] == 40000
    assert drivers["system_instruction_per_turn"] == 500
    assert drivers["system_instruction_cumulative"] == 5000


# ------------------------------------------------------------------ alerts


def test_prompt_alert_fires_once_and_only_once():
    """Alerts must not fire on every subsequent turn once the threshold is
    crossed, or the user learns to ignore them."""
    model = CostModel(warn_prompt_tokens_per_turn=10000)

    assert model.check_alerts(turn(0, prompt=5000), 5000) == []

    first = model.check_alerts(turn(1, prompt=12000), 17000)
    assert len(first) == 1
    assert first[0].kind == "prompt_tokens"

    again = model.check_alerts(turn(2, prompt=15000), 32000)
    assert again == []


def test_rent_ratio_alert():
    model = CostModel(warn_rent_ratio=0.90)
    t = TurnUsage(turn_index=5, prompt_tokens=9500, response_tokens=500, total_tokens=10000)
    alerts = model.check_alerts(t, 50000)

    assert [a.kind for a in alerts] == ["rent_ratio"]


def test_session_total_alert():
    model = CostModel(warn_session_total=100000)
    assert model.check_alerts(turn(0, prompt=100), 99999) == []
    assert [a.kind for a in model.check_alerts(turn(1, prompt=100), 100000)] == ["session_total"]


def test_alerts_disabled_when_thresholds_are_none():
    model = CostModel()
    t = TurnUsage(turn_index=0, prompt_tokens=10**9, response_tokens=1, total_tokens=10**9)
    assert model.check_alerts(t, 10**9) == []


def test_multiple_alerts_can_fire_on_the_same_turn():
    model = CostModel(
        warn_prompt_tokens_per_turn=1000,
        warn_rent_ratio=0.5,
        warn_session_total=1000,
    )
    t = TurnUsage(turn_index=0, prompt_tokens=5000, response_tokens=100, total_tokens=5100)
    kinds = {a.kind for a in model.check_alerts(t, 5100)}

    assert kinds == {"prompt_tokens", "rent_ratio", "session_total"}


def test_history_is_retained_for_the_growth_chart():
    model = CostModel()
    for i in range(4):
        model.observe_turn(turn(i, prompt=1000 * (i + 1)))

    assert [g.prompt_tokens for g in model.history] == [1000, 2000, 3000, 4000]


def test_cost_estimate_is_none_without_a_price_table():
    """Dollar figures are opt-in: a stale rate card is worse than no number."""
    assert CostModel().estimate_cost(turn(0, prompt=1000)) is None
