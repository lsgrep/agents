import pytest

from agentlab import budget as bg
from agentlab.budget import LoopShape


def shape(turns=40, **kw):
    base = dict(system_tokens=1_200, tool_tokens=6_000, prompt_tokens=300,
                assistant_tokens=250, result_tokens=1_400)
    base.update(kw)
    return LoopShape(turns=turns, **base)


def test_the_loop_bills_quadratically_not_linearly():
    # Doubling the turns roughly quadruples the input bill. This is the single
    # fact people are most surprised by, so it gets the plainest test.
    twenty = bg.loop_input_tokens(shape(turns=20))
    forty = bg.loop_input_tokens(shape(turns=40))
    assert forty / twenty > 3.4


def test_the_closed_form_matches_summing_turn_by_turn():
    s = shape(turns=17)
    by_hand = sum(s.fixed_prefix + i * s.per_turn_growth for i in range(s.turns))
    assert bg.loop_input_tokens(s) == by_hand


def test_most_of_the_bill_is_resent_transcript_not_the_prompt():
    s = shape(turns=40)
    fixed = s.turns * s.fixed_prefix
    assert bg.loop_input_tokens(s) > 4 * fixed


def test_caching_is_a_big_win_and_still_leaves_you_a_curve():
    cold40 = bg.run_cost(shape(turns=40), cached=False)
    warm40 = bg.run_cost(shape(turns=40), cached=True)
    assert warm40.usd < cold40.usd / 4          # it is a very big win

    # The token counts are untouched — only the rate they bill at changed.
    assert warm40.billed_tokens == cold40.input_tokens

    # And doubling the turns still costs far more than double, because the
    # term caching made cheap is exactly the quadratic one.
    warm20 = bg.run_cost(shape(turns=20), cached=True)
    assert warm40.usd / warm20.usd > 2.4


def test_the_cached_dollar_curve_steepens_as_the_run_grows():
    # Caching flattens the curve at moderate lengths because the write and
    # output terms are linear. The quadratic reasserts itself: the doubling
    # ratio climbs back toward the uncached one rather than settling at 2.
    def doubling(n):
        return bg.run_cost(shape(turns=2 * n), cached=True).usd / bg.run_cost(shape(turns=n), cached=True).usd

    assert doubling(20) < doubling(40) < doubling(80)
    assert doubling(80) > 3.0

    # ...and the read tokens themselves — the quadratic term — quadruple.
    reads = lambda n: bg.run_cost(shape(turns=n), cached=True).cache_read_tokens  # noqa: E731
    assert reads(80) / reads(40) > 3.5


def test_cached_and_uncached_account_for_the_same_tokens():
    s = shape(turns=25)
    cold = bg.run_cost(s, cached=False)
    warm = bg.run_cost(s, cached=True)
    assert warm.billed_tokens == cold.input_tokens


def test_a_cold_cache_costs_you_the_saving_which_is_why_hit_rate_is_measured():
    s = shape(turns=30)
    perfect = bg.run_cost(s, cached=True, cache_hit_rate=1.0)
    leaky = bg.run_cost(s, cached=True, cache_hit_rate=0.5)
    assert leaky.usd > perfect.usd * 2


def test_caching_does_not_pay_on_a_run_too_short_to_reuse_the_write():
    one = shape(turns=1)
    assert bg.run_cost(one, cached=True).usd > bg.run_cost(one, cached=False).usd


def test_bounding_the_context_is_the_only_lever_that_changes_the_shape():
    # Long enough for the cap to bite several times.
    s = shape(turns=120)
    unbounded = bg.run_cost(s, cached=True)
    bounded = bg.bounded_run_cost(s, cap=60_000)
    assert bounded.usd < unbounded.usd
    assert bounded.peak_context <= 60_000 < unbounded.peak_context


def test_bounding_a_run_that_barely_reaches_the_cap_costs_more_than_it_saves():
    # The compaction fires once, near the end, rewrites the whole prefix into
    # a fresh cache entry, and saves two turns of slightly cheaper reads. This
    # is a real and common way to ship a context strategy that loses money.
    s = shape(turns=35)
    assert bg.peak_context(s) > 60_000  # the cap does fire
    assert bg.bounded_run_cost(s, cap=60_000).usd > bg.run_cost(s, cached=True).usd


def test_there_is_a_crossover_and_it_is_worth_knowing_where():
    cheaper = [n for n in range(25, 81, 5)
               if bg.bounded_run_cost(shape(turns=n), cap=60_000).usd
               < bg.run_cost(shape(turns=n), cached=True).usd]
    assert cheaper and min(cheaper) > 40  # bounding only starts paying past ~40 turns here


def test_peak_context_is_what_has_to_fit_not_the_total():
    s = shape(turns=40)
    assert bg.peak_context(s) < bg.loop_input_tokens(s)
    assert bg.peak_context(s) == s.fixed_prefix + 39 * s.per_turn_growth


def test_tool_schemas_are_a_per_request_tax():
    assert bg.tool_overhead(30) == 30 * 380
    assert bg.context_share(bg.tool_overhead(150), "claude-haiku-4-5") > 0.25


def test_prices_are_a_dated_snapshot_that_says_so():
    assert bg.VERIFIED_ON
    assert bg.staleness(today="2026-06-25") == 1
    assert bg.price("claude-opus-5").cache_read == pytest.approx(0.5)
    assert bg.price("claude-opus-5").cache_write("1h") == pytest.approx(10.0)
    with pytest.raises(KeyError):
        bg.price("some-model-you-did-not-add")


def test_zero_turn_runs_do_not_divide_by_zero():
    assert bg.run_cost(shape(turns=0)).usd == 0.0


def test_bad_caps_are_rejected_rather_than_producing_a_plausible_number():
    with pytest.raises(ValueError):
        bg.bounded_run_cost(shape(), cap=100)
    with pytest.raises(ValueError):
        bg.bounded_run_cost(shape(), cap=50_000, keep_ratio=1.5)
