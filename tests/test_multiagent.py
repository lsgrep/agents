import pytest

from agentlab.budget import LoopShape
from agentlab.multiagent import (
    conjunctive_success,
    derive_fanout,
    disjunctive_success,
    fanout_tokens,
    optimal_fanout,
    single_agent_tokens,
    when_to_fan_out,
)

SHAPE = LoopShape(1_200, 6_000, 300, 250, 1_400, turns=48)


def test_fanning_out_divides_the_quadratic_term():
    one = single_agent_tokens(SHAPE)
    four = fanout_tokens(SHAPE, 4)["total"]
    assert four < one / 2


def test_each_subagent_gets_its_own_much_smaller_context():
    split = fanout_tokens(SHAPE, 4)
    assert split["peak_context_per_agent"] < SHAPE.fixed_prefix + SHAPE.per_turn_growth * 47


def test_conjunctive_fanout_multiplies_failure():
    # Five subagents that each work 95% of the time is a 77% system.
    assert conjunctive_success(0.95, 5) == pytest.approx(0.7738, abs=1e-4)
    assert conjunctive_success(0.95, 5) < 0.95


def test_disjunctive_fanout_divides_failure():
    assert disjunctive_success(0.95, 5) > 0.9999


def test_the_same_diagram_has_opposite_reliability_depending_on_topology():
    assert conjunctive_success(0.9, 4) < 0.9 < disjunctive_success(0.9, 4)


def test_token_cost_alone_always_says_more_workers_which_is_why_it_is_the_wrong_metric():
    rows = optimal_fanout(SHAPE)["rows"]
    assert rows[-1]["total"] < rows[0]["total"]


def test_cost_per_success_has_a_real_optimum_because_reliability_decays():
    result = optimal_fanout(SHAPE, p_worker=0.95, conjunctive=True)
    assert 1 < result["best_n"] < result["max_n_divisible"]
    assert result["best_usd_per_success"] < result["single_usd_per_success"]


def test_work_that_is_not_divisible_caps_the_fanout_regardless_of_economics():
    result = optimal_fanout(SHAPE, max_workers=48, min_turns_per_worker=12)
    assert result["max_n_divisible"] == 4


def test_the_checklist_says_no_to_the_cases_people_actually_bring():
    sequential = when_to_fan_out(read_heavy=True, parallelisable=False, shared_state=False, turns=40)
    assert not sequential["fan_out"] and "sequential" in sequential["reasons"][0]

    shared = when_to_fan_out(read_heavy=True, parallelisable=True, shared_state=True, turns=40)
    assert not shared["fan_out"]
    assert any("cannot see each other" in r for r in shared["reasons"])

    short = when_to_fan_out(read_heavy=True, parallelisable=True, shared_state=False, turns=4)
    assert not short["fan_out"]


def test_the_checklist_says_yes_to_breadth_first_read_heavy_work():
    ok = when_to_fan_out(read_heavy=True, parallelisable=True, shared_state=False,
                         turns=40, budget_multiplier=10)
    assert ok["fan_out"]


def test_the_derivation_flags_the_reliability_cost_of_a_conjunctive_split():
    d = derive_fanout(SHAPE, n_workers=4, p_worker=0.95, conjunctive=True)
    assert not d.all_checks_passed  # the reliability check fails, loudly
    assert "multiplies failure" in str(d)


def test_a_disjunctive_split_passes_the_same_check():
    assert derive_fanout(SHAPE, n_workers=4, p_worker=0.95, conjunctive=False).all_checks_passed


def test_a_single_worker_is_rejected_rather_than_silently_meaning_something_else():
    with pytest.raises(ValueError):
        fanout_tokens(SHAPE, 0)
