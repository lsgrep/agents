import pytest

from agentlab.reliability import horizon_success
from agentlab.sim import (
    Agent,
    Harness,
    Task,
    effective_p,
    horizon_sweep,
    lever_sweep,
    simulate,
    success_rate,
)


def test_a_run_is_deterministic_given_a_seed():
    a, b = simulate(seed=7), simulate(seed=7)
    assert a.summary == b.summary


def test_short_tasks_beat_the_independent_bound_because_a_wrong_step_is_survivable():
    # The p**n bound assumes one bad step kills the run. Real agents just take
    # another step, so at short horizons they do *better* than the bound.
    measured = success_rate(n_runs=200, task=Task(n_required=6))
    assert measured > horizon_success(Agent().p_step, 6)


def test_long_tasks_fall_off_a_cliff_the_bound_does_not_predict():
    # And then the feedback loop takes over and they do far worse.
    assert success_rate(n_runs=200, task=Task(n_required=34)) < 0.05


def test_the_curve_crosses_the_bound_and_that_crossing_is_the_lesson():
    rows = horizon_sweep(lengths=range(4, 41, 4), n_runs=200)
    gaps = [r["gap"] for r in rows]           # bound - measured
    assert gaps[0] < 0                        # early: measured is better
    assert gaps[-1] > 0                       # late: measured is far worse
    assert any(a < 0 <= b for a, b in zip(gaps, gaps[1:]))  # it crosses exactly once


def test_success_decays_monotonically_with_task_length():
    rows = horizon_sweep(lengths=range(4, 41, 6), n_runs=300)
    rates = [r["measured"] for r in rows]
    assert all(a >= b - 0.02 for a, b in zip(rates, rates[1:]))


def test_context_rot_lowers_the_step_probability_as_the_context_fills():
    agent = Agent()
    empty = effective_p(agent, 0, 60_000, confused=False)
    full = effective_p(agent, 60_000, 60_000, confused=False)
    assert empty == pytest.approx(agent.p_step)
    assert full < empty * 0.6


def test_a_flailing_agent_gets_worse_which_is_what_makes_it_a_spiral():
    agent = Agent()
    calm = effective_p(agent, 30_000, 60_000, confused=False)
    flailing = effective_p(agent, 30_000, 60_000, confused=True)
    assert flailing == pytest.approx(calm * agent.confusion_penalty)


def test_clearing_old_results_rescues_the_run_and_halves_the_bill():
    # The headline result of the whole simulator: the model did not change.
    # Only the harness did.
    unmanaged = simulate(task=Task(n_required=24), harness=Harness(), seed=1)
    managed = simulate(task=Task(n_required=24), harness=Harness(clear_after=6), seed=1)
    rates = {r["harness"]: r for r in lever_sweep(n_runs=200, n_required=24)}
    assert rates["clear old results"]["success_rate"] > rates["no management"]["success_rate"] + 0.15
    assert rates["clear old results"]["mean_input_tokens"] < rates["no management"]["mean_input_tokens"] * 0.7
    assert managed.summary["context_high_water"] < unmanaged.summary["context_high_water"]


def test_a_circuit_breaker_trades_success_for_bounded_cost_rather_than_adding_success():
    # Worth an honest test: the breaker aborts runs that would have recovered.
    # It is a cost control, not a quality lever, and confusing the two leads
    # people to ship it and wonder why the pass rate dropped.
    rates = {r["harness"]: r for r in lever_sweep(n_runs=200, n_required=24)}
    plain, broken = rates["no management"], rates["circuit breaker"]
    assert broken["mean_input_tokens"] < plain["mean_input_tokens"]
    assert broken["success_rate"] <= plain["success_rate"] + 0.02


def test_the_context_cap_is_a_real_stopping_condition():
    run = simulate(task=Task(n_required=60), harness=Harness(max_steps=200, context_cap=30_000), seed=2)
    assert run.summary["stop_reason"] == "context_exhausted"
    assert not run.success


def test_the_simulator_emits_the_same_metric_names_as_a_real_trace():
    from agentlab.trace import METRICS

    summary = simulate(seed=0).summary
    assert set(METRICS).issubset(summary)


def test_a_perfect_agent_with_no_rot_finishes_every_time():
    perfect = Agent(p_step=1.0, p_tool_error=0.0, rot=0.0)
    assert success_rate(n_runs=50, agent=perfect, task=Task(n_required=20)) == 1.0
