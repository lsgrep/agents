"""Context management: the four strategies, and what each one actually costs.

`agentlab.budget` proves that an unmanaged agent loop bills quadratically and
that only bounding the context changes that. This module is about the part
that arithmetic cannot tell you: **what you lose when you bound it.**

There are four strategies, and the interesting thing is that they fail
differently rather than better and worse:

- **keep everything** — perfect recall, quadratic cost, and a context that
  eventually degrades on its own (see `agentlab.sim`).
- **clear old tool results** — cheap and simple. Loses old detail *silently
  and completely*. Fine when old results are genuinely dead; catastrophic
  when step 30 needed something from step 4.
- **compact** — summarise the transcript into a shorter one. Keeps the gist,
  drops specifics, and the specifics are usually the identifiers, the exact
  error text and the numbers. A summary of a debugging session is a story
  about debugging, not the stack trace.
- **handles** — do not put the data in the transcript at all. Put a reference:
  a file path, a row id, a query. The agent re-reads what it needs when it
  needs it. Cost grows with what is *used*, not what was *seen*.

The last one is the one that changes the shape of the problem, and it is why
"just-in-time context" and "the filesystem is the memory" keep turning up in
production agent designs. The others manage a growing transcript. This one
stops the transcript from being where the data lives.

`compare()` runs all four against the same synthetic task and reports recall
against tokens, so the tradeoff is a table rather than an opinion.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# A task whose facts are needed later
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """One piece of information the agent saw at `step` and may need at the end."""

    id: str
    step: int
    tokens: int
    specific: bool = True  #: an identifier or exact value — the kind a summary drops


def make_task(n_steps: int = 40, n_facts: int = 20, result_tokens: int = 1_400,
              seed: int = 0) -> list:
    """A run where facts arrive throughout and are needed at the end.

    Deliberately adversarial in the way real work is: the facts you need are
    not the recent ones. If they were, `clear_after` would always win and
    context engineering would be a solved problem.
    """
    rng = random.Random(seed)
    facts = []
    for i in range(n_facts):
        step = rng.randrange(1, n_steps + 1)
        facts.append(
            Fact(
                id=f"fact-{i:02d}",
                step=step,
                tokens=rng.randrange(result_tokens // 3, result_tokens),
                specific=rng.random() < 0.6,
            )
        )
    return sorted(facts, key=lambda f: f.step)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    strategy: str
    recall: float  #: fraction of needed facts still reachable at the end
    input_tokens: int  #: total billed input across the run
    peak_context: int
    extra_calls: int = 0  #: retrievals the strategy forced
    note: str = ""

    def __str__(self) -> str:
        return (
            f"{self.strategy:<22} recall {self.recall:>6.1%}  "
            f"input {self.input_tokens:>10,}  peak {self.peak_context:>8,}  "
            f"+{self.extra_calls} calls"
        )


def _walk(facts: Sequence[Fact], n_steps: int, start_tokens: int, assistant_tokens: int,
          on_step) -> tuple:
    """Shared machinery: step through a run, letting `on_step` mutate the context."""
    ctx = start_tokens
    live: list = []  # (fact, tokens_in_context)
    billed = peak = 0
    extra = 0
    for step in range(1, n_steps + 1):
        for fact in (f for f in facts if f.step == step):
            live.append([fact, fact.tokens])
            ctx += fact.tokens
        ctx += assistant_tokens
        ctx, live, added = on_step(step, ctx, live)
        extra += added
        billed += ctx
        peak = max(peak, ctx)
    return billed, peak, live, extra


def keep_everything(facts, n_steps=40, start_tokens=7_500, assistant_tokens=250) -> Outcome:
    def on_step(step, ctx, live):
        return ctx, live, 0

    billed, peak, live, extra = _walk(facts, n_steps, start_tokens, assistant_tokens, on_step)
    return Outcome(
        "keep everything", 1.0, billed, peak, extra,
        "perfect recall, quadratic cost — and the context that makes `sim` fall over",
    )


def clear_old_results(facts, keep_last: int = 6, n_steps=40, start_tokens=7_500,
                      assistant_tokens=250) -> Outcome:
    def on_step(step, ctx, live):
        if len(live) > keep_last:
            dropped = live[:-keep_last]
            ctx -= sum(t for _, t in dropped)
            live = live[-keep_last:]
        return ctx, live, 0

    billed, peak, live, extra = _walk(facts, n_steps, start_tokens, assistant_tokens, on_step)
    kept = {f.id for f, _ in live}
    recall = len(kept) / len(facts) if facts else 1.0
    return Outcome(
        f"clear (keep {keep_last})", recall, billed, peak, extra,
        "cheapest thing that works, and it loses old detail completely and silently",
    )


def compact(facts, threshold: int = 25_000, keep_ratio: float = 0.3, fidelity: float = 0.5,
            n_steps=40, start_tokens=7_500, assistant_tokens=250, seed: int = 0) -> Outcome:
    """Summarise when the context crosses `threshold`.

    `fidelity` is the honest parameter: the probability a *specific* fact
    survives summarisation. Set it from your own measurement — summarise a real
    transcript, then check how many identifiers, error strings and numbers made
    it through. It is usually lower than people guess, and it is lowest for
    exactly the facts that matter.

    Note the threshold has to sit *below* the run's natural peak or nothing
    happens — a compaction trigger set above where your runs actually reach is
    a very common way to ship a feature that never once fires.
    """
    rng = random.Random(seed)
    compactions = 0

    def on_step(step, ctx, live):
        nonlocal compactions
        if ctx < threshold:
            return ctx, live, 0
        compactions += 1
        new_live = []
        for fact, _ in live:
            # A summary keeps themes more reliably than it keeps specifics.
            p = fidelity if fact.specific else min(1.0, fidelity + 0.4)
            if rng.random() < p:
                new_live.append([fact, max(20, fact.tokens // 8)])  # one line in the summary
        kept_tokens = sum(t for _, t in new_live)
        ctx = start_tokens + min(int(threshold * keep_ratio), kept_tokens)
        return ctx, new_live, 1  # the summarisation is itself a model call

    billed, peak, live, extra = _walk(facts, n_steps, start_tokens, assistant_tokens, on_step)
    reachable = {f.id for f, _ in live}
    recall = len(reachable) / len(facts) if facts else 1.0
    return Outcome(
        "compact", recall, billed, peak, extra,
        f"keeps the gist across {compactions} compactions. Drops identifiers, exact errors "
        "and numbers — the specifics",
    )


def handles(facts, keep_last: int = 4, handle_tokens: int = 40, reread_fraction: float = 0.35,
            n_steps=40, start_tokens=7_500, assistant_tokens=250, seed: int = 0) -> Outcome:
    """Keep a reference, not the payload. Re-read on demand.

    This is the strategy that changes the *shape*: context grows with the number
    of things seen (tiny — one handle each) rather than with their size, and
    recall is bounded by whether the agent knows to look, not by what fits.

    The cost is real and shows up as `extra_calls`: a re-read is a round trip,
    and a round trip is a turn, and a turn resends the transcript.
    `reread_fraction` is how much of the run genuinely needs a payload back.
    """
    rng = random.Random(seed)

    def on_step(step, ctx, live):
        added = 0
        for entry in live:
            if entry[1] > handle_tokens:  # demote the payload to a reference
                ctx -= entry[1] - handle_tokens
                entry[1] = handle_tokens
        for entry in live[-keep_last:]:  # re-materialise what this step needs
            if entry[1] == handle_tokens and rng.random() < reread_fraction / max(1, keep_last):
                ctx += entry[0].tokens
                entry[1] = entry[0].tokens
                added += 1
        return ctx, live, added

    billed, peak, live, extra = _walk(facts, n_steps, start_tokens, assistant_tokens, on_step)
    return Outcome(
        "handles (+re-read)", 1.0, billed, peak, extra,
        "everything stays addressable; context grows with what you use, not what you saw",
    )


def compare(facts: Sequence[Fact] | None = None, n_steps: int = 40, **kw) -> list:
    """All four on the same task. Read the recall column and the token column together.

    Neither column alone picks a winner, which is the entire point: the cheapest
    strategy and the one that remembers your data are different strategies, and
    which you want depends on whether step 30 needs what step 4 saw.
    """
    facts = list(facts) if facts is not None else make_task(n_steps=n_steps)
    return [
        keep_everything(facts, n_steps=n_steps, **kw),
        clear_old_results(facts, n_steps=n_steps, **kw),
        compact(facts, n_steps=n_steps, **kw),
        handles(facts, n_steps=n_steps, **kw),
    ]


# --------------------------------------------------------------------------
# Notes: the durable half
# --------------------------------------------------------------------------


@dataclass
class Notebook:
    """An agent's own notes — the cheapest durable memory there is.

    A note is a fact the agent chose to keep, written in its own words, at a
    fraction of the tokens of the thing it came from. It survives compaction
    (it is short and it is early), it survives clearing (you re-inject it), and
    it survives the session (it is a file).

    The failure mode is that the agent has to *decide* to write one, and that
    decision is made with the information it has at the time. Notes are lossy
    in the same direction as summaries — they just cost far less.
    """

    entries: list = field(default_factory=list)
    max_tokens: int = 2_000

    def write(self, text: str, tokens: int | None = None) -> bool:
        from .budget import estimate_tokens

        cost = tokens if tokens is not None else estimate_tokens(text)
        if self.tokens + cost > self.max_tokens:
            return False
        self.entries.append((text, cost))
        return True

    @property
    def tokens(self) -> int:
        return sum(c for _, c in self.entries)

    def render(self) -> str:
        return "\n".join(f"- {t}" for t, _ in self.entries)
