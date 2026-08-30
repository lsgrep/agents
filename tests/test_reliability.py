import math

import pytest

from agentlab import reliability as rel


def test_the_headline_number_a_95_percent_agent_is_a_36_percent_agent():
    # This single line is the reason the rest of this repo exists.
    assert rel.horizon_success(0.95, 20) == pytest.approx(0.3585, abs=1e-4)


def test_a_ninety_percent_target_at_95_percent_steps_buys_you_two_steps():
    assert rel.steps_at_reliability(0.95, 0.9) == pytest.approx(2.05, abs=0.01)


def test_a_hundred_step_task_demands_accuracy_nobody_measures_at():
    # 99.9% per step. You cannot even *establish* that rate on a 200-case eval.
    assert rel.required_step_accuracy(100, 0.9) == pytest.approx(0.99895, abs=1e-5)


def test_retries_help_but_only_as_far_as_the_verifier_sees():
    perfect = rel.with_retries(0.9, 3)
    blind = rel.with_imperfect_verifier(0.9, 3, p_detect=0.5)
    half = rel.with_imperfect_verifier(0.9, 3, p_detect=1.0)
    assert perfect == pytest.approx(0.999)
    assert half == pytest.approx(perfect, abs=1e-9)  # a perfect verifier recovers the ideal case
    assert blind < perfect  # an unseen failure is committed, not retried


def test_checkpoints_do_not_change_p_step_they_cap_what_a_failure_destroys():
    plain = rel.horizon_success(0.9, 24)
    segmented = rel.checkpoint_horizon(0.9, 24, segment=4)
    assert segmented > plain


def test_pass_at_k_answers_capability_and_pass_pow_k_answers_reliability():
    # Same agent. The first number is the demo, the second is production.
    assert rel.pass_at_k(10, 6, 1) == pytest.approx(0.6)
    assert rel.pass_at_k(10, 6, 5) > 0.95
    assert rel.pass_pow_k(0.6, 5) == pytest.approx(0.07776)


def test_pass_at_k_is_one_when_failures_cannot_fill_the_draw():
    assert rel.pass_at_k(10, 9, 5) == 1.0


def test_empirical_pass_pow_k_counts_only_cases_that_never_flake():
    always = [True] * 5
    flaky = [True, True, False, True, True]
    never = [False] * 5
    assert rel.pass_pow_k_empirical([always, always], 5) == 1.0
    assert rel.pass_pow_k_empirical([always, never], 5) == 0.5
    assert rel.pass_pow_k_empirical([flaky], 5) == 0.0  # 4 of 5 is not 5 of 5


def test_wilson_stays_inside_zero_and_one_where_the_normal_interval_does_not():
    lo, hi = rel.wilson_interval(20, 20)
    assert 0 <= lo <= hi <= 1.0
    assert lo > 0.8  # and it is not vacuous


def test_a_forty_case_eval_cannot_see_a_five_point_regression():
    assert rel.detectable_difference(40, 0.7) > 0.05
    assert rel.samples_for_half_width(0.7, 0.05) > 300


def test_mcnemar_uses_only_the_disagreements():
    # The 100 cases both versions passed carry no information about which is
    # better; an unpaired comparison spends data pretending they do.
    baseline = [True] * 100 + [True] * 8 + [False] * 2
    candidate = [True] * 100 + [False] * 8 + [True] * 2
    improved, regressed, p = rel.mcnemar(baseline, candidate)
    assert (improved, regressed) == (2, 8)
    assert p < 0.15
    assert rel.mcnemar([True] * 10, [True] * 10)[2] == 1.0


def test_mcnemar_refuses_unpaired_input():
    with pytest.raises(ValueError):
        rel.mcnemar([True, False], [True])


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_probabilities_are_validated_rather_than_silently_wrapped(bad):
    with pytest.raises(ValueError):
        rel.horizon_success(bad, 3)


def test_the_derivation_agrees_with_the_functions_it_documents():
    d = rel.derive_horizon(0.95, 20, target=0.9)
    assert d.value == pytest.approx(rel.required_step_accuracy(20, 0.9))
    assert not d.all_checks_passed  # 95% per step does not reach 90% over 20 steps
    assert math.isclose(rel.horizon_success(0.95, 20), 0.3585, abs_tol=1e-3)
