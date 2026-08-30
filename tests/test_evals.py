import pytest

from agentlab.evals import (
    Case,
    Gate,
    Report,
    bootstrap_ci,
    evaluate,
    judge_agreement,
    reliability_report,
    score,
)
from agentlab.trace import ToolCall, Trace, Turn


def trace_of(text="the answer is 42", tools=(), steps=1, stop="end_turn"):
    turns = [Turn(index=i + 1, tool_calls=[ToolCall(name=t) for t in tools] if i == 0 else [])
             for i in range(steps)]
    return Trace(turns=turns, final_text=text, stop_reason=stop)


def test_the_right_answer_by_the_wrong_route_does_not_pass():
    # An agent that deletes the account and then reports the right number has
    # not passed. Outcome-only scoring says it has.
    case = Case("c1", "how many?", expect=r"42", forbids=("delete_account",))
    s = score(case, trace_of(tools=["delete_account"]))
    assert s.outcome and not s.trajectory and not s.passed
    assert "forbidden" in s.reasons[0]


def test_a_required_tool_that_was_never_called_fails_the_trajectory():
    case = Case("c1", "look it up", expect=r"42", requires=("search",))
    assert not score(case, trace_of(tools=["guess"])).passed


def test_running_out_of_steps_is_a_failure_not_a_missing_result():
    case = Case("c1", "go", expect=r"42")
    s = score(case, trace_of(stop="max_steps"))
    assert not s.outcome and "ran out" in " ".join(s.reasons)


def test_a_step_budget_is_part_of_passing():
    case = Case("c1", "go", expect=r"42", max_steps=2)
    assert score(case, trace_of(steps=2)).passed
    assert not score(case, trace_of(steps=9)).passed


def test_the_report_separates_right_answers_from_passing_runs():
    cases = [Case(f"c{i}", "go", expect=r"42", forbids=("cheat",)) for i in range(4)]
    traces = [trace_of(), trace_of(tools=["cheat"]), trace_of(), trace_of()]
    report = evaluate(cases, lambda c: traces[int(c.id[1:])])
    assert report.outcome_rate == 1.0
    assert report.pass_rate == 0.75
    assert "the wrong way" in str(report)


def test_the_report_states_its_own_confidence_interval():
    report = Report(scores=[score(Case("c", "x", expect="42"), trace_of()) for _ in range(20)])
    lo, hi = report.interval()
    assert lo < 1.0 and hi == 1.0  # 20/20 is not proof of 100%


def test_high_agreement_can_still_be_a_useless_judge():
    # 92% agreement sounds gate-worthy. Kappa says it is not: on a set that is
    # mostly pass, agreeing is easy.
    human = [True] * 85 + [False] * 15
    judge = [True] * 85 + [True] * 8 + [False] * 7
    result = judge_agreement(judge, human)
    assert result["agreement"] > 0.9
    assert result["kappa"] < 0.8
    assert result["verdict"] == "ranking only"


def test_a_judge_that_always_says_pass_scores_zero_kappa():
    human = [True] * 90 + [False] * 10
    result = judge_agreement([True] * 100, human)
    assert result["agreement"] == 0.9
    assert result["kappa"] == pytest.approx(0.0, abs=1e-9)
    assert "not usable" in result["verdict"]


def test_the_judges_false_passes_are_reported_separately_because_they_cost_more():
    human = [True] * 5 + [False] * 5
    judge = [True] * 5 + [True] * 3 + [False] * 2
    assert judge_agreement(judge, human)["false_pass"] == 3


def test_a_small_gate_reports_the_regression_it_could_not_see():
    baseline = [True] * 28 + [False] * 12
    candidate = [True] * 26 + [False] * 14
    result = Gate().check(baseline, candidate)
    assert result["verdict"] == "no change visible"
    assert result["underpowered"]
    assert result["blind_spot"] > 0.15
    assert "cannot see" in result["note"]


def test_a_big_enough_gate_calls_a_real_regression():
    baseline = [True] * 200
    candidate = [True] * 170 + [False] * 30
    result = Gate().check(baseline, candidate)
    assert result["verdict"] == "REGRESSION"
    assert result["significant"] and result["regressed"] == 30


def test_the_gate_is_paired_and_says_so_when_it_cannot_be():
    with pytest.raises(ValueError):
        Gate().check([True, False], [True])


def test_pass_at_one_and_pass_pow_k_come_apart_on_a_flaky_agent():
    calls = {"n": 0}

    def flaky(case):
        calls["n"] += 1
        return trace_of(text="42" if calls["n"] % 2 else "wrong")

    result = reliability_report([Case("c", "go", expect="42")], flaky, k=4)
    assert result["pass@1"] == pytest.approx(0.5)
    assert result[f"pass^{4}"] == 0.0
    assert result["flaky_cases"] == 1


def test_bootstrap_gives_an_interval_for_scores_that_are_not_pass_fail():
    lo, hi = bootstrap_ci([0.6, 0.7, 0.8, 0.9, 0.55, 0.72], n_boot=500, seed=1)
    assert lo < 0.72 < hi
