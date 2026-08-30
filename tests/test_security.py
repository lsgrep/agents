from agentlab.loop import (
    ModelResponse,
    ScriptedModel,
    Tool,
    ToolRegistry,
    run,
    text_block,
    tool_use_block,
)
from agentlab.security import (
    EXFIL,
    INJECTION_PROBES,
    PRIVATE,
    UNTRUSTED,
    analyze,
    infer_capabilities,
    probe_harness,
    taint,
)


def trifecta_surface():
    return [
        Tool("read_customer_record", "Read a customer's private account details from the database."),
        Tool("fetch_url", "Fetch a web page and return its text.", tainted_output=True),
        Tool("send_email", "Send an email to any address.", read_only=False),
    ]


def test_all_three_legs_in_one_agent_is_critical():
    result = analyze(trifecta_surface())
    assert result.has_trifecta
    assert result.critical[0].severity == "critical"
    assert "exfiltrate" in result.critical[0].message


def test_two_legs_is_a_warning_that_names_the_missing_one():
    two = [t for t in trifecta_surface() if t.name != "send_email"]
    result = analyze(two)
    assert not result.has_trifecta
    risks = [r for r in result.risks if r.kind == "two_of_three"]
    assert risks and "exfiltration" in risks[0].message


def test_breaking_a_leg_clears_the_critical_finding():
    # The architectural fix, tested: remove outbound reach and the shape is safe
    # even though the model and the prompt are unchanged.
    safe = [t for t in trifecta_surface() if t.name != "send_email"]
    assert not analyze(safe).has_trifecta


def test_untrusted_content_reaching_a_destructive_tool_is_its_own_finding():
    surface = [Tool("fetch_url", "Fetch a web page.", tainted_output=True),
               Tool("delete_account", "Permanently delete an account.",
                    read_only=False, destructive=True)]
    kinds = {r.kind for r in analyze(surface).risks}
    assert "untrusted_to_destructive" in kinds


def test_a_destructive_tool_the_harness_cannot_gate_is_flagged():
    surface = [Tool("purge_data", "Purge all records.", read_only=True, destructive=True)]
    assert "ungated_destructive" in {r.kind for r in analyze(surface).risks}


def test_capabilities_can_be_declared_explicitly_rather_than_guessed():
    tool = Tool("innocuous_name", "Does a thing.")
    assert infer_capabilities(tool) == set()
    tool.capabilities = {PRIVATE, EXFIL}
    assert infer_capabilities(tool) == {PRIVATE, EXFIL}


def test_the_heuristic_errs_toward_flagging():
    assert UNTRUSTED in infer_capabilities(Tool("browse_web", "Browse a URL."))
    assert PRIVATE in infer_capabilities(Tool("get_secret", "Read a secret."))
    assert EXFIL in infer_capabilities(Tool("post_webhook", "Send a webhook."))


def test_taint_flags_the_outbound_call_that_follows_untrusted_input():
    surface = trifecta_surface()
    registry = ToolRegistry([
        Tool(t.name, t.description, fn=lambda **kw: "ok",
             read_only=t.read_only, destructive=t.destructive, tainted_output=t.tainted_output)
        for t in surface
    ])
    model = ScriptedModel([
        ModelResponse([tool_use_block("1", "read_customer_record", {})], "tool_use"),
        ModelResponse([tool_use_block("2", "fetch_url", {"url": "http://example.test"})], "tool_use"),
        ModelResponse([tool_use_block("3", "send_email", {"to": "a@b.test"})], "tool_use"),
        ModelResponse([text_block("done")], "end_turn"),
    ])
    events = taint(run(model, registry, "help"), surface)
    assert [e.tool for e in events] == ["fetch_url", "send_email"]
    assert "1 step after" in events[1].message


def test_an_outbound_call_before_any_untrusted_input_is_not_flagged():
    surface = trifecta_surface()
    registry = ToolRegistry([
        Tool(t.name, t.description, fn=lambda **kw: "ok", tainted_output=t.tainted_output)
        for t in surface
    ])
    model = ScriptedModel([
        ModelResponse([tool_use_block("1", "send_email", {"to": "a@b.test"})], "tool_use"),
        ModelResponse([text_block("done")], "end_turn"),
    ])
    assert taint(run(model, registry, "help"), surface) == []


def test_the_probe_corpus_reports_what_got_through_rather_than_a_score():
    result = probe_harness(lambda text: "ignore" in text.lower())
    assert result["n"] == len(INJECTION_PROBES)
    assert "direct_override" in result["blocked"]
    assert result["passed_through"]
    assert "speed bump" in result["note"]


def test_a_guard_that_blocks_everything_still_gets_the_architecture_warning():
    result = probe_harness(lambda text: True)
    assert result["passed_through"] == []
    assert "architecture is the control" in result["note"]


def test_a_tool_with_two_capabilities_is_named_once_not_twice():
    # read_ticket is both untrusted input and (arguably) private data. Listing
    # it twice makes the finding read like a bug in the analyzer.
    surface = [
        Tool("read_ticket", "Read a support ticket, including the customer's messages.",
             tainted_output=True),
        Tool("read_customer_record", "Read the customer's private account details."),
        Tool("post_reply", "Post a public reply on the ticket.", read_only=False),
    ]
    named = analyze(surface).critical[0].tools
    assert len(named) == len(set(named))
