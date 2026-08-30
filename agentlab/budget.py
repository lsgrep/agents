"""Context economics: what an agent loop actually costs, and why.

The single most useful fact in this repo, and the one nobody derives before
their first bill:

    **A naive agent loop bills quadratically in the number of turns.**

Every turn resends the whole transcript. Turn 1 sends the prefix; turn 20 sends
the prefix plus nineteen turns of assistant text and tool results. Summed over
a run of `n` turns that is

    total_input = n * P0 + (a + t) * n(n-1)/2

— linear in the fixed prefix, **quadratic** in the growing tail. A 40-turn run
does not cost twice a 20-turn run. It costs about four times as much.

Prompt caching does not fix this, and it is worth being precise about what
it does do, because the half-true version ("caching makes it linear") gets
repeated a lot. Caching leaves the *token* counts identical and moves the
quadratic term onto the read rate, which is 0.1x input. It is the single
highest-leverage change you can make — typically an 80%+ cut. But the curve is
still a curve: the terms it makes cheap are the quadratic ones, and the terms
that stay at full price (the per-turn cache write, the output) are linear. So
the dollar curve is *flattened at moderate run lengths and steepens again as
the run grows* — doubling 20 turns to 40 costs about 2.5x, doubling 80 to 160
costs about 3.2x, converging back toward the uncached 3.8x.

The only things that change the shape are the ones that bound the context —
compaction, clearing tool results, keeping data behind handles instead of in
the transcript. That is the whole argument for context engineering, and it is
arithmetic rather than taste. It is also not free, and it does not pay on short
runs: `bounded_run_cost` will happily show you a compaction that fires once,
near the end, and costs more than it saved.

Everything here is closed form and provider-neutral. `PRICES` is a snapshot
you maintain; see the note on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .derive import Derivation, Worksheet

MTOK = 1_000_000

# --------------------------------------------------------------------------
# Prices — a snapshot you maintain, not a source of truth
# --------------------------------------------------------------------------

#: Update this when you re-check the pricing page, and `staleness()` will stop
#: complaining. It cannot re-verify for you.
VERIFIED_ON = "2026-06-24"

#: Cache multipliers relative to that model's base input price. These are
#: pricing *policy* rather than per-model numbers, which is why they live
#: apart from the table: a 5-minute cache write costs 1.25x base input, a
#: 1-hour write 2x, and a read 0.1x. Re-check them with everything else.
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.00
CACHE_READ = 0.10


@dataclass(frozen=True)
class Price:
    """USD per million tokens, plus the context window you have to fit in."""

    name: str
    input: float
    output: float
    context: int

    def cache_write(self, ttl: str = "5m") -> float:
        return self.input * (CACHE_WRITE_1H if ttl == "1h" else CACHE_WRITE_5M)

    @property
    def cache_read(self) -> float:
        return self.input * CACHE_READ


#: A snapshot. API pricing moves, intro rates expire, and a lab that quotes a
#: stale number confidently is worse than one that quotes none. Re-check before
#: you put a cost figure in front of anyone, and edit this table rather than
#: the code that reads it.
PRICES = {
    "claude-opus-5": Price("claude-opus-5", 5.00, 25.00, 1_000_000),
    "claude-sonnet-5": Price("claude-sonnet-5", 2.00, 10.00, 1_000_000),
    "claude-haiku-4-5": Price("claude-haiku-4-5", 1.00, 5.00, 200_000),
}


def staleness(today: str | None = None) -> int:
    """Days since `VERIFIED_ON`. Print it before you quote a cost."""
    from datetime import date

    y, m, d = (int(x) for x in VERIFIED_ON.split("-"))
    now = date.fromisoformat(today) if today else date.today()
    return (now - date(y, m, d)).days


def price(model: str) -> Price:
    if model not in PRICES:
        raise KeyError(f"{model!r} is not in the snapshot; add it to PRICES with a source")
    return PRICES[model]


# --------------------------------------------------------------------------
# Counting tokens without an API call
# --------------------------------------------------------------------------

#: Bytes per token for ordinary English prose. JSON, code and non-Latin text
#: are all denser than this. It is a planning heuristic, not a measurement —
#: `agentlab.providers.count_tokens` calls the real tokenizer.
CHARS_PER_TOKEN = 3.8


def estimate_tokens(text: str) -> int:
    """A planning-grade token estimate. Never use it in a billing reconciliation."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopShape:
    """The five numbers that determine what a run costs.

    Read them off a real trace (`agentlab.trace.Trace.shape()` does it for
    you) rather than guessing — `result_tokens` in particular is almost always
    larger than people expect, because a tool that returns a file, a page or a
    query result puts all of it in the transcript forever.
    """

    system_tokens: int  #: system prompt, stable across the run
    tool_tokens: int  #: every tool schema, sent on every single request
    prompt_tokens: int  #: the user's opening message
    assistant_tokens: int  #: per turn: thinking + text + the tool_use block
    result_tokens: int  #: per turn: what the tool handed back
    turns: int

    @property
    def fixed_prefix(self) -> int:
        """P0 — resent, in full, on every request of the run."""
        return self.system_tokens + self.tool_tokens + self.prompt_tokens

    @property
    def per_turn_growth(self) -> int:
        """How much longer the transcript gets each turn."""
        return self.assistant_tokens + self.result_tokens


@dataclass(frozen=True)
class Cost:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    usd: float
    peak_context: int

    @property
    def billed_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __str__(self) -> str:
        return (
            f"${self.usd:,.4f}  "
            f"in {self.input_tokens:,} / read {self.cache_read_tokens:,} / "
            f"write {self.cache_write_tokens:,} / out {self.output_tokens:,}  "
            f"peak context {self.peak_context:,}"
        )


def loop_input_tokens(shape: LoopShape) -> int:
    """Total input tokens billed across a naive run. The quadratic, in one line.

    `n*P0 + g*n(n-1)/2`, where `g` is the per-turn growth. The second term is
    the one that surprises people.
    """
    n, p0, g = shape.turns, shape.fixed_prefix, shape.per_turn_growth
    return n * p0 + g * (n * (n - 1)) // 2


def peak_context(shape: LoopShape) -> int:
    """Longest single request in the run — what has to fit in the window."""
    return shape.fixed_prefix + shape.per_turn_growth * max(0, shape.turns - 1)


def run_cost(shape: LoopShape, model: str = "claude-opus-5", cached: bool = False,
             cache_hit_rate: float = 1.0, ttl: str = "5m") -> Cost:
    """What one run of this shape costs, with and without prompt caching.

    Without caching every input token bills at the input rate. With caching,
    turn `i` reads everything that was already in the transcript at turn `i-1`
    and writes only the delta, so the quadratic term moves onto the read rate.

    `cache_hit_rate` is the honest knob: caches expire (5 minutes by default),
    and an agent that waits on a slow tool, a human approval, or a queue comes
    back to a cold prefix and pays full price for it. Set it from measurement —
    `usage.cache_read_input_tokens` over what you expected to be read — not
    from hope.
    """
    p = price(model)
    n, p0, g = shape.turns, shape.fixed_prefix, shape.per_turn_growth
    out = n * shape.assistant_tokens
    if n <= 0:
        return Cost(0, 0, 0, 0, 0.0, 0)

    if not cached:
        inp = loop_input_tokens(shape)
        usd = inp * p.input / MTOK + out * p.output / MTOK
        return Cost(inp, out, 0, 0, usd, peak_context(shape))

    # Turn 1 writes the fixed prefix. Turn i (>=2) reads the transcript as of
    # turn i-1 and writes the delta that turn i-1 appended.
    read_total = sum(p0 + (i - 2) * g for i in range(2, n + 1))
    write_total = p0 + (n - 1) * g
    hit = max(0.0, min(1.0, cache_hit_rate))
    read_hit = int(read_total * hit)
    read_miss = read_total - read_hit  # billed as ordinary input
    usd = (
        read_hit * p.cache_read / MTOK
        + read_miss * p.input / MTOK
        + write_total * p.cache_write(ttl) / MTOK
        + out * p.output / MTOK
    )
    return Cost(read_miss, out, read_hit, write_total, usd, peak_context(shape))


def bounded_run_cost(shape: LoopShape, cap: int, keep_ratio: float = 0.3,
                     model: str = "claude-opus-5", cached: bool = True) -> Cost:
    """The same run, with the transcript bounded at `cap` tokens.

    When the context reaches `cap`, it is compacted down to `cap * keep_ratio`
    and growth resumes from there. This is what turns the quadratic back into
    a line: past the first compaction the average request size stops climbing.

    Compaction is not free — the summary costs a call and it *loses* things,
    which is what `agentlab.context` measures. This function prices it; it does
    not tell you whether the run still works afterwards.
    """
    if cap <= shape.fixed_prefix:
        raise ValueError("cap must leave room for the fixed prefix")
    if not 0 < keep_ratio < 1:
        raise ValueError("keep_ratio must be in (0, 1)")
    p = price(model)
    ctx = shape.fixed_prefix
    read_total = write_total = plain_input = 0
    for turn in range(1, shape.turns + 1):
        if turn == 1:
            write_total += ctx
        elif cached:
            read_total += ctx - shape.per_turn_growth
            write_total += shape.per_turn_growth
        else:
            plain_input += ctx
        ctx += shape.per_turn_growth
        if ctx > cap:
            ctx = int(cap * keep_ratio) + shape.fixed_prefix
            write_total += ctx  # the compacted prefix has to be cached afresh
    out = shape.turns * shape.assistant_tokens
    usd = (
        read_total * p.cache_read / MTOK
        + write_total * p.cache_write() / MTOK
        + plain_input * p.input / MTOK
        + out * p.output / MTOK
    )
    return Cost(plain_input, out, read_total, write_total, usd, min(cap, peak_context(shape)))


def tool_overhead(n_tools: int, tokens_per_tool: int = 380) -> int:
    """Tokens spent describing tools you might use, on every single request.

    `tokens_per_tool` is a middling MCP-style schema with a description and a
    handful of typed parameters. Measure your own — the spread is wide, and a
    verbose schema is a tax you pay per turn, per run, forever.
    """
    if n_tools < 0 or tokens_per_tool < 0:
        raise ValueError("counts must be >= 0")
    return n_tools * tokens_per_tool


def context_share(tokens: int, model: str = "claude-opus-5") -> float:
    """What fraction of the window something occupies before work begins."""
    return tokens / price(model).context


# --------------------------------------------------------------------------
# With the working shown
# --------------------------------------------------------------------------


def derive_loop_tokens(shape: LoopShape) -> Derivation:
    n, p0, g = shape.turns, shape.fixed_prefix, shape.per_turn_growth
    fixed, growing = n * p0, g * (n * (n - 1)) // 2
    total = fixed + growing
    naive = n * p0  # what people assume they are paying
    return (
        Derivation(
            "Input tokens billed across one run",
            "total_input = n*P0 + g*n(n-1)/2      P0 = system + tools + prompt,  g = assistant + result",
        )
        .given("system_tokens", shape.system_tokens, "tok", "your system prompt")
        .given("tool_tokens", shape.tool_tokens, "tok", "every schema, every request")
        .given("prompt_tokens", shape.prompt_tokens, "tok")
        .given("assistant_tokens", shape.assistant_tokens, "tok/turn", "thinking + text + tool_use")
        .given("result_tokens", shape.result_tokens, "tok/turn", "what the tool returned")
        .given("turns", n, "turns", "count them in a real trace")
        .step("P0", f"{shape.system_tokens} + {shape.tool_tokens} + {shape.prompt_tokens}", p0, "tok")
        .step("g", f"{shape.assistant_tokens} + {shape.result_tokens}", g, "tok/turn")
        .step("fixed term", f"{n} * {p0}", fixed, "tok")
        .step("growing term", f"{g} * {n}*{n - 1}/2", growing, "tok")
        .step("total input", f"{fixed} + {growing}", total, "tok")
        .step("peak context", f"{p0} + {g}*{max(0, n - 1)}", peak_context(shape), "tok")
        .check(
            "the tail dominates",
            growing > fixed,
            f"{growing / max(total, 1):.0%} of your input bill is resent transcript",
        )
        .check(
            "peak fits the window",
            context_share(peak_context(shape)) < 1.0,
            f"{context_share(peak_context(shape)):.1%} of a 1M window",
        )
        .says(
            f"This run bills {total:,} input tokens, not the {naive:,} the prefix suggests — "
            f"the transcript is resent every turn, so cost grows with the square of the turn count."
        )
    )


def derive_cache_savings(shape: LoopShape, model: str = "claude-opus-5",
                         cache_hit_rate: float = 1.0) -> Derivation:
    cold = run_cost(shape, model, cached=False)
    warm = run_cost(shape, model, cached=True, cache_hit_rate=cache_hit_rate)
    saving = 1 - warm.usd / cold.usd if cold.usd else 0.0
    p = price(model)
    return (
        Derivation(
            f"What caching is worth on this run ({model})",
            "read = 0.10x input   write(5m) = 1.25x input",
        )
        .given("input rate", p.input, "$/Mtok", f"snapshot, verified {VERIFIED_ON}")
        .given("cache read rate", p.cache_read, "$/Mtok", "0.1x input")
        .given("cache write rate", p.cache_write(), "$/Mtok", "1.25x input, 5-minute TTL")
        .given("cache hit rate", cache_hit_rate, "%", "measure it; do not assume 1.0")
        .step("uncached", f"{cold.input_tokens:,} in + {cold.output_tokens:,} out", cold.usd, "$")
        .step("cached", f"{warm.cache_read_tokens:,} read + {warm.cache_write_tokens:,} write", warm.usd, "$")
        .step("saving", f"1 - {warm.usd:.4f}/{cold.usd:.4f}", saving, "%")
        .check("caching pays here", warm.usd < cold.usd, "it does not on very short runs — the write costs 1.25x")
        .check(
            "output is not the bill",
            cold.output_tokens * p.output < cold.input_tokens * p.input,
            "on agent loops the input side usually dominates, which is the opposite of chat",
        )
        .says(
            f"Caching takes this run from ${cold.usd:,.2f} to ${warm.usd:,.2f} ({saving:.0%}). "
            "It divides the quadratic by ten; it does not remove it."
        )
    )


def worksheet(shape: LoopShape, model: str = "claude-opus-5", cap: int | None = None) -> Worksheet:
    """The whole cost chain for one run shape: tokens, caching, and a bound."""
    ws = Worksheet(f"Run economics — {shape.turns} turns on {model}")
    ws.add(derive_loop_tokens(shape))
    ws.add(derive_cache_savings(shape, model))
    if cap:
        warm = run_cost(shape, model, cached=True)
        bounded = bounded_run_cost(shape, cap, model=model)
        ws.add(
            Derivation("Bounding the context", f"compact at {cap:,} tokens, keep 30%")
            .given("cap", cap, "tok", "your compaction trigger")
            .step("unbounded, cached", f"peak {warm.peak_context:,} tok", warm.usd, "$")
            .step("bounded, cached", f"peak {bounded.peak_context:,} tok", bounded.usd, "$")
            .check("the bound pays", bounded.usd <= warm.usd, "on short runs it does not — you paid to summarise nothing")
            .says(
                "Compaction is the only lever that changes the *shape* of the cost curve. "
                "Caching changes its slope."
            )
        )
    return ws
