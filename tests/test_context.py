
from agentlab.context import (
    Notebook,
    clear_old_results,
    compact,
    compare,
    handles,
    keep_everything,
    make_task,
)


def test_the_task_puts_needed_facts_early_not_just_recently():
    facts = make_task(n_steps=40, n_facts=20, seed=0)
    assert min(f.step for f in facts) < 10  # otherwise clear_after always wins


def test_keeping_everything_has_perfect_recall_and_the_biggest_bill():
    facts = make_task()
    rows = {o.strategy: o for o in compare(facts)}
    assert rows["keep everything"].recall == 1.0
    assert rows["keep everything"].input_tokens == max(o.input_tokens for o in rows.values())


def test_clearing_is_cheap_and_loses_old_facts_completely_and_silently():
    facts = make_task()
    out = clear_old_results(facts, keep_last=6)
    assert out.recall < 0.5
    assert out.input_tokens < keep_everything(facts).input_tokens


def test_compaction_keeps_the_gist_and_drops_the_specifics():
    facts = make_task()
    out = compact(facts, threshold=25_000, fidelity=0.5)
    assert 0.4 < out.recall < 1.0            # better than clearing, not lossless
    assert out.extra_calls >= 1              # the summary is itself a model call


def test_a_compaction_threshold_above_the_runs_peak_never_fires():
    # A very common way to ship a context strategy that does nothing at all.
    facts = make_task()
    never = compact(facts, threshold=10_000_000)
    assert never.recall == 1.0
    assert never.extra_calls == 0
    assert never.input_tokens == keep_everything(facts).input_tokens


def test_handles_keep_everything_addressable_at_the_lowest_token_cost():
    # The strategy that changes the shape: context grows with what you use,
    # not with what you saw.
    facts = make_task()
    out = handles(facts)
    assert out.recall == 1.0
    assert out.input_tokens < clear_old_results(facts).input_tokens
    assert out.extra_calls > 0               # and it is not free — re-reads are turns


def test_the_comparison_is_a_real_tradeoff_with_no_single_winner():
    rows = {o.strategy: o for o in compare()}
    cheapest = min(rows.values(), key=lambda o: o.input_tokens)
    assert cheapest.strategy.startswith("handles")
    assert rows["clear (keep 6)"].recall < rows["compact"].recall < 1.0
    assert rows["clear (keep 6)"].input_tokens < rows["keep everything"].input_tokens


def test_the_comparison_is_deterministic_so_a_lesson_can_quote_it():
    assert [(o.strategy, o.recall, o.input_tokens) for o in compare()] == \
           [(o.strategy, o.recall, o.input_tokens) for o in compare()]


def test_a_notebook_is_bounded_and_refuses_to_overflow():
    nb = Notebook(max_tokens=50)
    assert nb.write("short note", tokens=20)
    assert nb.write("another", tokens=20)
    assert not nb.write("one too many", tokens=20)
    assert nb.tokens == 40
    assert "short note" in nb.render()
