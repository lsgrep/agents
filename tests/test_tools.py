from agentlab.loop import Tool
from agentlab.tools import distractors, error, lint, report, schema_tokens, surface_tokens


def kinds(findings):
    return {f.kind for f in findings}


def test_interchangeable_verbs_over_the_same_object_are_flagged_loudly():
    findings = lint([Tool("get_status", "Return the current status of the job."),
                     Tool("fetch_status", "Fetch the current status of the job.")])
    assert "ambiguous_pair" in kinds(findings)
    assert findings[0].severity == "high"


def test_the_same_verb_over_different_objects_is_not_a_finding():
    # get_order and get_customer are told apart by the object. Flagging these
    # would bury the real collisions under noise.
    findings = lint([Tool("get_order", "Look up a single order by its identifier."),
                     Tool("get_customer", "Look up a single customer by its identifier.")])
    assert "ambiguous_pair" not in kinds(findings)


def test_the_control_surface_produces_no_findings():
    # collisions=False is the control: vary surface *size* without also
    # varying surface *quality*.
    assert lint(distractors(10, collisions=False)) == []
    assert lint(distractors(24, collisions=False), budget_share=1.0) == []


def test_a_realistic_surface_grows_collisions_on_its_own():
    # This is the finding, not a defect in the generator: an API that accumulates
    # a tool at a time accumulates near-duplicates faster than it accumulates
    # tools.
    def pairs(n):
        return len([f for f in lint(distractors(n), budget_share=1.0)
                    if f.kind == "ambiguous_pair"])

    assert pairs(10) < pairs(80) < pairs(150)


def test_a_large_surface_is_flagged_for_what_it_costs_every_request():
    findings = lint(distractors(60), budget_share=0.001)
    cost = [f for f in findings if f.kind == "surface_cost"]
    assert cost and "every request" in cost[0].message


def test_an_unbounded_collection_tool_is_flagged():
    findings = lint([Tool("search_logs", "Search all log entries and return the results.",
                          {"type": "object", "properties": {"q": {"type": "string"}}})])
    assert "unbounded_output" in kinds(findings)


def test_a_bounded_collection_tool_is_not():
    findings = lint([Tool("search_logs", "Search all log entries and return the results.",
                          {"type": "object", "properties": {"q": {"type": "string"},
                                                            "limit": {"type": "integer"}}})])
    assert "unbounded_output" not in kinds(findings)


def test_a_thin_description_is_flagged_because_it_is_the_only_selection_signal():
    assert "thin_description" in kinds(lint([Tool("do_thing", "Does it.")]))


def test_a_free_text_parameter_where_an_enum_was_meant_is_flagged():
    findings = lint([Tool("list_items", "List every item, paged.",
                          {"type": "object", "properties": {
                              "sort": {"type": "string"},
                              "limit": {"type": "integer"}}})])
    assert "untyped_choice" in kinds(findings)


def test_an_enum_parameter_is_not_flagged():
    findings = lint([Tool("list_items", "List every item, paged.",
                          {"type": "object", "properties": {
                              "sort": {"type": "string", "enum": ["asc", "desc"]},
                              "limit": {"type": "integer"}}})])
    assert "untyped_choice" not in kinds(findings)


def test_a_destructive_tool_the_harness_cannot_gate_is_flagged():
    findings = lint([Tool("drop_table", "Delete a table permanently and irreversibly.",
                          read_only=True, destructive=True)])
    assert "ungated_destructive" in kinds(findings)


def test_findings_come_back_worst_first():
    findings = lint([Tool("get_x", "Get x."), Tool("fetch_x", "Fetch the x value.")])
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_surface_cost_scales_with_the_number_of_tools():
    assert surface_tokens(distractors(20)) > 2 * surface_tokens(distractors(10)) * 0.8
    assert schema_tokens(Tool("t", "d").schema()) > 0


def test_the_report_collapses_a_repeated_pattern_instead_of_repeating_it():
    surface = [Tool(f"search_{n}", f"Search all {n} entries and return the results.",
                    {"type": "object", "properties": {"q": {"type": "string"}}})
               for n in ("orders", "users", "items", "logs", "events")]
    text = report(surface)
    assert "and 2 more unbounded_output" in text


def test_an_error_message_tells_the_model_what_to_do_next():
    msg = error("search failed", "status must be known", "Retry with a valid status",
                valid=["open", "closed"])
    assert "open, closed" in msg and "Retry" in msg and msg.startswith("Error:")


def test_asking_for_more_distractors_than_the_vocabulary_allows_is_an_error():
    # Not a hang: the generator needs distinct names, and past a point there
    # are none left. This is the failure mode that ate an afternoon.
    import pytest

    big = distractors(150)
    assert len({t.name for t in big}) == 150
    with pytest.raises(ValueError, match="can only generate"):
        distractors(10_000)
    with pytest.raises(ValueError, match="can only generate"):
        distractors(100, collisions=False)  # one verb per noun caps it much lower


def test_generated_surfaces_are_deterministic_for_a_seed():
    assert [t.name for t in distractors(20, seed=3)] == [t.name for t in distractors(20, seed=3)]
