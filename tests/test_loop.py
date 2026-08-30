import pytest

from agentlab.loop import (
    LoopInvariantError,
    ModelResponse,
    PolicyModel,
    ScriptedModel,
    Tool,
    ToolRegistry,
    assistant,
    check_transcript,
    run,
    text_block,
    tool_result_block,
    tool_use_block,
    user,
)


def adder():
    return Tool("add", "Add two numbers together and return the sum.",
                {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                 "required": ["a", "b"]},
                fn=lambda a, b: a + b)


def test_the_loop_ends_when_the_model_stops_asking_for_tools():
    m = ScriptedModel([ModelResponse([text_block("done")], "end_turn")])
    trace = run(m, ToolRegistry([adder()]), "hi")
    assert trace.stop_reason == "end_turn"
    assert trace.final_text == "done"
    assert trace.tool_calls == 0


def test_parallel_tool_calls_come_back_in_exactly_one_user_message():
    # Splitting them across several messages silently teaches the model to
    # stop making parallel calls. This is invariant 3, and it is the one that
    # fails without ever raising anything.
    m = ScriptedModel([
        ModelResponse([tool_use_block("a", "add", {"a": 1, "b": 2}),
                       tool_use_block("b", "add", {"a": 3, "b": 4})], "tool_use"),
        ModelResponse([text_block("3 and 7")], "end_turn"),
    ])
    trace = run(m, ToolRegistry([adder()]), "add")
    messages = trace.meta["messages"]
    results = [msg for msg in messages
               if any(b.get("type") == "tool_result" for b in msg["content"])]
    assert len(results) == 1
    assert len(results[0]["content"]) == 2
    assert results[0]["role"] == "user"


def test_a_tool_that_raises_becomes_a_recoverable_error_not_a_crash():
    boom = Tool("boom", "Always fails, for testing.", fn=lambda: 1 / 0)
    m = ScriptedModel([
        ModelResponse([tool_use_block("x", "boom", {})], "tool_use"),
        ModelResponse([text_block("I will try something else")], "end_turn"),
    ])
    trace = run(m, ToolRegistry([boom]), "go")
    assert trace.tool_errors == 1
    assert "ZeroDivisionError" in trace.calls[0].result
    assert trace.stop_reason == "end_turn"  # the run survived it


def test_a_hallucinated_tool_name_is_answered_with_the_names_that_exist():
    m = ScriptedModel([
        ModelResponse([tool_use_block("x", "subtract", {})], "tool_use"),
        ModelResponse([text_block("ok")], "end_turn"),
    ])
    trace = run(m, ToolRegistry([adder()]), "go")
    assert trace.calls[0].is_error
    assert "add" in trace.calls[0].result  # it is told what it *could* have called


def test_max_steps_is_a_circuit_breaker_that_actually_fires():
    forever = ScriptedModel([ModelResponse([tool_use_block(f"i{i}", "add", {"a": 1, "b": 1})],
                                           "tool_use") for i in range(50)])
    trace = run(forever, ToolRegistry([adder()]), "loop", max_steps=5)
    assert trace.stop_reason == "max_steps"
    assert trace.steps == 5


def test_the_token_budget_stops_a_run_that_the_step_limit_would_not():
    def policy(messages, tools):
        return ModelResponse([tool_use_block(f"t{len(messages)}", "add", {"a": 1, "b": 1})], "tool_use")

    trace = run(PolicyModel(policy), ToolRegistry([adder()]), "go",
                max_steps=100, max_tokens_budget=2_000)
    assert trace.stop_reason == "budget"
    assert trace.steps < 100


def test_repeated_identical_calls_are_counted_because_that_is_the_derailment_signal():
    m = ScriptedModel([
        ModelResponse([tool_use_block(f"t{i}", "add", {"a": 1, "b": 1})], "tool_use")
        for i in range(4)
    ] + [ModelResponse([text_block("done")], "end_turn")])
    trace = run(m, ToolRegistry([adder()]), "go")
    assert trace.tool_calls == 4
    assert trace.distinct_calls == 1
    assert trace.repeat_calls == 3


def test_the_registry_refuses_two_tools_with_the_same_name():
    reg = ToolRegistry([adder()])
    with pytest.raises(ValueError):
        reg.add(adder())


def test_a_tool_schema_carries_only_what_goes_on_the_wire():
    # read_only / destructive are for the harness, not the model.
    schema = Tool("t", "d", destructive=True).schema()
    assert set(schema) == {"name", "description", "input_schema"}


def test_check_transcript_catches_an_unanswered_tool_use():
    bad = [user("hi"), assistant([tool_use_block("a", "add", {})])]
    with pytest.raises(LoopInvariantError):
        check_transcript(bad)


def test_check_transcript_catches_results_split_across_two_messages():
    bad = [
        user("hi"),
        assistant([tool_use_block("a", "add", {}), tool_use_block("b", "add", {})]),
        user([tool_result_block("a", "1")]),
        user([tool_result_block("b", "2")]),
    ]
    with pytest.raises(LoopInvariantError):
        check_transcript(bad)


def test_check_transcript_catches_a_result_for_a_call_that_never_happened():
    bad = [user("hi"), assistant([tool_use_block("a", "add", {})]),
           user([tool_result_block("ghost", "1")])]
    with pytest.raises(LoopInvariantError):
        check_transcript(bad)


def test_a_well_formed_transcript_passes():
    good = [user("hi"), assistant([tool_use_block("a", "add", {})]),
            user([tool_result_block("a", "3")]), assistant("done")]
    check_transcript(good)  # does not raise


def test_the_on_step_hook_can_rewrite_the_transcript_between_turns():
    # This hook is where every context-management strategy in this repo lives.
    seen = []

    def hook(turn, messages):
        seen.append(len(messages))

    m = ScriptedModel([
        ModelResponse([tool_use_block("a", "add", {"a": 1, "b": 1})], "tool_use"),
        ModelResponse([text_block("2")], "end_turn"),
    ])
    run(m, ToolRegistry([adder()]), "go", on_step=hook)
    assert seen == [3]  # prompt + assistant + results
