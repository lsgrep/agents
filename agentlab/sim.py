"""The doom loop, simulated.

`agentlab.reliability` gives the optimistic bound: independent steps, `p ** n`,
graceful decay. Real agents do worse than that bound, and they do worse in a
specific, reproducible shape that is worth being able to recognise before you
meet it in production at 2am.

The mechanism is a feedback loop with three parts, and none of them are exotic:

1. A step fails. The failure — an error message, a wrong result, a retry — is
   **appended to the transcript**, because the transcript is the state.
2. A longer transcript is a worse transcript. Attention is finite; recall and
   instruction-following degrade as the context fills. Call it context rot.
3. A worse transcript makes the next step more likely to fail. Go to 1.

That is positive feedback, and positive feedback does not degrade gracefully.
It holds up, holds up, holds up, and then falls over — which is why agent
reliability reports so often read "worked fine in testing". Short tasks never
enter the loop. The failure is a property of the *horizon*, not of the model.

This module is deliberately a simulation and not a benchmark. It runs in
milliseconds on a CPU with no API key, so you can sweep a thousand
configurations and *see the shape*, and it emits the same metric names as
`agentlab.trace`, so the dashboard you write here plots your real runs
unchanged. The absolute numbers are made up. The shape is the lesson, and the
shape is what transfers.

The most useful thing you can do with it is turn a lever off and watch the
cliff move: `context_cap`, `clear_after`, `max_consecutive_failures`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# The knobs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Agent:
    """The model's behaviour, reduced to five numbers you can measure.

    `p_step` and `p_tool_error` come out of a real trace. `rot` is the one that
    needs your own measurement — run your eval at 10K, 100K and 500K tokens of
    filled context and fit it. Everyone's is different, and everyone's is
    worse than they expect.
    """

    p_step: float = 0.95  #: P(a step is productive) with an empty context
    p_tool_error: float = 0.04  #: P(the tool itself fails: bad args, 500, timeout)
    rot: float = 0.45  #: fraction of p_step lost at a completely full context
    rot_power: float = 2.0  #: >1 means the damage is back-loaded, which matches reports
    confusion_after: int = 3  #: consecutive failures before the agent starts flailing
    confusion_penalty: float = 0.5  #: multiplier on p_step once it is flailing


@dataclass(frozen=True)
class Task:
    """What finishing looks like."""

    n_required: int = 12  #: productive steps needed
    assistant_tokens: int = 250
    result_tokens: int = 1_400
    error_tokens: int = 220  #: a failure is cheaper in tokens — but not free


@dataclass(frozen=True)
class Harness:
    """Your code. Every field here is a lever you actually control.

    This is the point of the whole module: the model is one row in `Agent`,
    and everything else on this page is something you decide.
    """

    max_steps: int = 60
    context_cap: int = 60_000  #: where the run is declared dead (or compacted)
    token_budget: int | None = None
    clear_after: int | None = None  #: keep only the last N tool results
    compact_at: float | None = None  #: fraction of cap that triggers compaction
    compact_to: float = 0.35  #: what fraction of the cap you compact down to
    max_consecutive_failures: int | None = None  #: circuit breaker
    start_tokens: int = 7_500  #: system + tools + prompt


@dataclass
class Run:
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.summary.get("success"))

    def __str__(self) -> str:
        s = self.summary
        return (
            f"{'DONE' if s['success'] else 'FAILED'} in {s['steps']} steps "
            f"({s['tool_errors']} errors, {s['repeat_calls']} repeats), "
            f"peak context {s['context_high_water']:,}, stopped on {s['stop_reason']}"
        )


def effective_p(agent: Agent, context_tokens: int, cap: int, confused: bool) -> float:
    """P(productive step) given how full the context is. The rot curve.

    `p_step * (1 - rot * fill**power)`, then halved again if the agent has
    started flailing. Fill past the cap keeps hurting — real runs do not stop
    degrading at a round number.
    """
    fill = max(0.0, context_tokens / cap) if cap else 0.0
    p = agent.p_step * (1.0 - agent.rot * min(fill, 1.5) ** agent.rot_power)
    if confused:
        p *= agent.confusion_penalty
    return max(0.0, min(1.0, p))


def simulate(agent: Agent = Agent(), task: Task = Task(), harness: Harness = Harness(),
             seed: int = 0) -> Run:
    """One run. Deterministic given `seed`, so a lesson can point at a specific one."""
    rng = random.Random(seed)
    ctx = harness.start_tokens
    progress = consecutive_failures = errors = repeats = 0
    recent: list = []  # tool results currently in context, for `clear_after`
    peak = ctx
    stop = "max_steps"
    rows: list = []

    for step in range(1, harness.max_steps + 1):
        confused = consecutive_failures >= agent.confusion_after
        p = effective_p(agent, ctx, harness.context_cap, confused)

        if rng.random() < agent.p_tool_error:
            outcome, _gained = "tool_error", 0
            ctx += task.assistant_tokens + task.error_tokens
            errors += 1
            consecutive_failures += 1
        elif rng.random() < p:
            outcome, _gained = "progress", 1
            ctx += task.assistant_tokens + task.result_tokens
            recent.append(task.result_tokens)
            progress += 1
            consecutive_failures = 0
        else:
            outcome, _gained = "wrong_step", 0
            ctx += task.assistant_tokens + task.result_tokens
            recent.append(task.result_tokens)
            consecutive_failures += 1
            if confused:
                repeats += 1  # a flailing agent re-calls what it already called

        peak = max(peak, ctx)
        rows.append(
            {
                "step": step,
                "outcome": outcome,
                "progress": progress,
                "p_effective": round(p, 4),
                "context_tokens": ctx,
                "consecutive_failures": consecutive_failures,
                "confused": confused,
                "repeat_calls": repeats,
                "tool_errors": errors,
            }
        )

        # --- the harness gets its turn ------------------------------------
        if harness.clear_after is not None and len(recent) > harness.clear_after:
            dropped = recent[: -harness.clear_after]
            recent = recent[-harness.clear_after :]
            ctx -= sum(dropped)
        if harness.compact_at is not None and ctx >= harness.context_cap * harness.compact_at:
            ctx = int(harness.context_cap * harness.compact_to) + harness.start_tokens
            recent = []

        if progress >= task.n_required:
            stop = "end_turn"
            break
        if (harness.max_consecutive_failures is not None
                and consecutive_failures >= harness.max_consecutive_failures):
            stop = "circuit_breaker"
            break
        if ctx >= harness.context_cap:
            stop = "context_exhausted"
            break
        if harness.token_budget is not None and ctx >= harness.token_budget:
            stop = "budget"
            break

    n_steps = len(rows)
    summary = {
        "steps": n_steps,
        "tool_calls": n_steps,
        "tool_errors": errors,
        "error_rate": round(errors / n_steps, 4) if n_steps else 0.0,
        "repeat_calls": repeats,
        "distinct_calls": n_steps - repeats,
        "input_tokens": sum(r["context_tokens"] for r in rows),
        "output_tokens": n_steps * task.assistant_tokens,
        "context_high_water": peak,
        "stop_reason": stop,
        "success": progress >= task.n_required,
        "progress": progress,
        "required": task.n_required,
    }
    return Run(rows, summary)


# --------------------------------------------------------------------------
# Sweeps — the point is the shape, so make it cheap to see one
# --------------------------------------------------------------------------


def success_rate(n_runs: int = 200, seed: int = 0, **kwargs) -> float:
    """Fraction of `n_runs` seeds that finish. This is pass^1, measured."""
    return sum(simulate(seed=seed + i, **kwargs).success for i in range(n_runs)) / n_runs


def horizon_sweep(lengths=range(2, 41, 2), n_runs: int = 200, agent: Agent = Agent(),
                  harness: Harness = Harness(), seed: int = 0) -> list:
    """Success against task length — the cliff.

    Plot it beside `reliability.horizon_success(p, n)` on the same axes. The
    analytic curve is the optimistic bound; the simulated one falls away from
    it, and the gap is everything this module is about.
    """
    from .reliability import horizon_success

    rows = []
    for n in lengths:
        measured = success_rate(
            n_runs=n_runs, seed=seed, agent=agent, task=Task(n_required=n), harness=harness
        )
        rows.append(
            {
                "n_required": n,
                "measured": measured,
                "independent_bound": horizon_success(agent.p_step, n),
                "gap": horizon_success(agent.p_step, n) - measured,
            }
        )
    return rows


def lever_sweep(n_runs: int = 200, n_required: int = 24, agent: Agent = Agent(),
                seed: int = 0) -> list:
    """The same task under four harnesses. This is the table worth memorising.

    Nothing about the model changes across these rows. Only your code does.
    """
    levers = [
        ("no management", Harness()),
        ("circuit breaker", Harness(max_consecutive_failures=4)),
        ("clear old results", Harness(clear_after=6)),
        ("clear + breaker", Harness(clear_after=6, max_consecutive_failures=4)),
        ("compaction", Harness(compact_at=0.7)),
    ]
    rows = []
    for name, harness in levers:
        runs = [simulate(agent=agent, task=Task(n_required=n_required), harness=harness, seed=seed + i)
                for i in range(n_runs)]
        rows.append(
            {
                "harness": name,
                "success_rate": sum(r.success for r in runs) / n_runs,
                "mean_steps": sum(r.summary["steps"] for r in runs) / n_runs,
                "mean_input_tokens": sum(r.summary["input_tokens"] for r in runs) // n_runs,
                "peak_context": max(r.summary["context_high_water"] for r in runs),
            }
        )
    return rows
