from agentlab.trace import METRICS, ToolCall, Trace, Turn, Usage, summarize


def build(calls, stop="end_turn", success=True):
    turns = []
    for i, batch in enumerate(calls):
        turns.append(Turn(index=i + 1,
                          tool_calls=[ToolCall(name=n, input=a, tokens=100) for n, a in batch],
                          usage=Usage(input_tokens=1_000, output_tokens=50),
                          context_tokens=1_000 * (i + 1)))
    return Trace(turns=turns, stop_reason=stop, success=success)


def test_identical_calls_with_identical_arguments_are_repeats():
    trace = build([[("search", {"q": "x"})], [("search", {"q": "x"})], [("search", {"q": "y"})]])
    assert trace.tool_calls == 3
    assert trace.distinct_calls == 2
    assert trace.repeat_calls == 1


def test_the_same_tool_with_different_arguments_is_not_a_repeat():
    trace = build([[("read", {"path": "a"})], [("read", {"path": "b"})]])
    assert trace.repeat_calls == 0


def test_an_a_b_a_b_cycle_is_detected():
    trace = build([[("a", {})], [("b", {})], [("a", {})], [("b", {})]])
    assert trace.longest_cycle() == 2


def test_a_straight_line_run_has_no_cycle():
    assert build([[("a", {})], [("b", {})], [("c", {})]]).longest_cycle() == 0


def test_error_rate_is_over_calls_not_turns():
    trace = build([[("a", {}), ("b", {})]])
    trace.turns[0].tool_calls[0].is_error = True
    assert trace.tool_errors == 1
    assert trace.error_rate == 0.5


def test_context_high_water_is_the_peak_not_the_last_value():
    # A run that compacted mid-way ends small but still had to fit the peak.
    trace = build([[("a", {})], [("b", {})]])
    assert trace.turns[1].context_tokens == 2_000
    trace.turns[1].context_tokens = 500          # ...then compaction happened
    assert trace.context_high_water == 1_000     # the peak, not the last value


def test_usage_adds_up_across_turns():
    usage = build([[("a", {})], [("b", {})]]).usage
    assert usage.input_tokens == 2_000 and usage.output_tokens == 100


def test_the_summary_carries_exactly_the_shared_metric_names():
    assert set(build([[("a", {})]]).summary()) == set(METRICS)


def test_a_trace_can_be_fitted_to_a_loop_shape_for_pricing():
    trace = build([[("a", {})], [("b", {})], [("c", {})]])
    trace.meta.update(system_tokens=800, tool_tokens=2_000)
    shape = trace.shape()
    assert shape.turns == 3
    assert shape.system_tokens == 800
    assert shape.result_tokens == 100


def test_summarize_reports_the_tail_not_just_the_mean():
    traces = [build([[("a", {})]] * n) for n in (1, 2, 3, 20)]
    out = summarize(traces)
    assert out["runs"] == 4
    assert out["p90_steps"] >= out["mean_steps"]
    assert out["peak_context"] == 20_000
    assert out["success_rate"] == 1.0


def test_summarize_counts_runs_that_hit_the_step_limit():
    traces = [build([[("a", {})]], stop="max_steps"), build([[("a", {})]])]
    assert summarize(traces)["hit_step_limit"] == 1


def test_summarize_of_nothing_is_empty_rather_than_an_exception():
    assert summarize([]) == {}
