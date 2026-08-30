"""Fan-out: when a second agent pays for itself, and when it just costs twice.

Multi-agent systems are usually justified as a *speed* play — do things in
parallel. That is the least interesting reason and often not even true, since
wall-clock is bounded by the slowest branch and by your rate limits.

The real argument is arithmetic, and it falls straight out of
`agentlab.budget`. One agent doing `m` steps bills roughly

    m * P0 + g * m² / 2

Split the same work across `n` subagents doing `m/n` steps each, and each one
has **its own transcript**. Total:

    n * (m/n) * P0 + n * g * (m/n)² / 2  =  m * P0 + g * m² / (2n)

The quadratic term is divided by `n`. The fixed prefix is multiplied by... one
(each subagent runs fewer turns, and the terms cancel) — except that in
practice each subagent carries its own system prompt and tool schemas, and the
orchestrator pays to brief them and to synthesise what comes back. So the real
shape is: **a quadratic you divide, plus a linear overhead you add.** That is
an optimisation problem with an interior optimum, not a slogan.

It also explains the number people quote from production deep-research systems:
roughly an order of magnitude more tokens than a single-agent chat. That is not
the split being inefficient. It is subagents doing far more total exploration
because they *can* — the context ceiling that limited one agent no longer binds.
You are buying breadth with money.

The second half is risk, and it goes the other way. Fan-out has a topology:

- **conjunctive** (every subagent must succeed — split a task into parts):
  `p ** n`. Fan-out *multiplies* your failure probability. Five subagents at
  95% is an 77% system.
- **disjunctive** (any subagent succeeding is enough — search the same question
  several ways): `1 - (1-p) ** n`. Fan-out *divides* it.

Same architecture diagram, opposite reliability. Knowing which one you have
drawn is most of the design review.
"""

from __future__ import annotations

from .budget import MTOK, LoopShape, loop_input_tokens, price
from .derive import Derivation


def single_agent_tokens(shape: LoopShape) -> int:
    return loop_input_tokens(shape)


def fanout_tokens(shape: LoopShape, n_workers: int, brief_tokens: int = 400,
                  report_tokens: int = 1_200, orchestrator_turns: int = 3) -> dict:
    """Input tokens for the same work split across `n_workers` subagents.

    Accounts for the three things that make fan-out cost more than the algebra
    suggests: each worker resends its own fixed prefix, the orchestrator writes
    a brief for each, and it reads a report back from each — and those reports
    land in the orchestrator's transcript, where they are resent every turn.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    per_worker_turns = max(1, round(shape.turns / n_workers))
    worker_shape = LoopShape(
        system_tokens=shape.system_tokens,
        tool_tokens=shape.tool_tokens,
        prompt_tokens=brief_tokens,
        assistant_tokens=shape.assistant_tokens,
        result_tokens=shape.result_tokens,
        turns=per_worker_turns,
    )
    workers = n_workers * loop_input_tokens(worker_shape)
    orch_shape = LoopShape(
        system_tokens=shape.system_tokens,
        tool_tokens=shape.tool_tokens,
        prompt_tokens=shape.prompt_tokens,
        assistant_tokens=shape.assistant_tokens,
        result_tokens=report_tokens * n_workers // max(1, orchestrator_turns),
        turns=orchestrator_turns + 1,
    )
    orchestrator = loop_input_tokens(orch_shape)
    return {
        "n_workers": n_workers,
        "worker_turns_each": per_worker_turns,
        "worker_tokens": workers,
        "orchestrator_tokens": orchestrator,
        "total": workers + orchestrator,
        "peak_context_per_agent": worker_shape.fixed_prefix
        + worker_shape.per_turn_growth * max(0, per_worker_turns - 1),
    }


def optimal_fanout(shape: LoopShape, max_workers: int = 12, p_worker: float = 0.95,
                   conjunctive: bool = True, min_turns_per_worker: int = 4,
                   model: str = "claude-opus-5", **kw) -> dict:
    """Sweep `n` and find where fan-out actually stops paying.

    Token cost alone is misleading: it falls monotonically in `n`, because the
    quadratic term is divided by `n` and the overhead grows slowly. Read that
    curve on its own and the answer is always "more workers", which is wrong
    and expensive.

    The number that has a genuine optimum is **cost per successful task** —
    tokens divided by P(system succeeds). Under conjunctive fan-out reliability
    decays as `p ** n`, so past a certain point you are buying cheaper runs that
    fail more often, and paying for the retries. That crossover is the real
    answer, and it is usually a small number.

    `min_turns_per_worker` is the other bound, and it is not economic: work
    stops being divisible. Forty-eight dependent steps do not become forty-eight
    independent agents, and a model that says otherwise is measuring an
    architecture nobody can build.
    """
    max_n = max(1, min(max_workers, shape.turns // max(1, min_turns_per_worker)))
    success = conjunctive_success if conjunctive else disjunctive_success
    rows = []
    for n in range(1, max_n + 1):
        split = fanout_tokens(shape, n, **kw)
        p = success(p_worker, n)
        usd = cost(split["total"], model)
        rows.append(
            {
                "n": n,
                **split,
                "p_success": p,
                "usd": usd,
                "usd_per_success": usd / p if p > 0 else float("inf"),
            }
        )
    cheapest = min(rows, key=lambda r: r["total"])
    best = min(rows, key=lambda r: r["usd_per_success"])
    return {
        "rows": rows,
        "max_n_divisible": max_n,
        "cheapest_n": cheapest["n"],
        "best_n": best["n"],
        "best_usd_per_success": best["usd_per_success"],
        "single_usd_per_success": rows[0]["usd_per_success"],
        "saving": 1 - best["usd_per_success"] / rows[0]["usd_per_success"]
        if rows[0]["usd_per_success"] else 0.0,
    }


def conjunctive_success(p_worker: float, n_workers: int) -> float:
    """Every worker must succeed. Fan-out multiplies risk: `p ** n`."""
    if not 0 <= p_worker <= 1:
        raise ValueError("p_worker must be in [0, 1]")
    return p_worker**n_workers


def disjunctive_success(p_worker: float, n_workers: int) -> float:
    """Any worker succeeding is enough. Fan-out divides risk: `1 - (1-p) ** n`."""
    if not 0 <= p_worker <= 1:
        raise ValueError("p_worker must be in [0, 1]")
    return 1 - (1 - p_worker) ** n_workers


def cost(tokens: int, model: str = "claude-opus-5") -> float:
    return tokens * price(model).input / MTOK


def derive_fanout(shape: LoopShape, n_workers: int = 4, p_worker: float = 0.95,
                  model: str = "claude-opus-5", conjunctive: bool = True) -> Derivation:
    single = single_agent_tokens(shape)
    split = fanout_tokens(shape, n_workers)
    p = (conjunctive_success if conjunctive else disjunctive_success)(p_worker, n_workers)
    topology = "conjunctive (all must succeed)" if conjunctive else "disjunctive (any will do)"
    return (
        Derivation(
            f"Fan-out to {n_workers} subagents — {topology}",
            "single: m*P0 + g*m²/2      split: m*P0 + g*m²/(2n) + briefs + reports",
        )
        .given("turns of work", shape.turns, "turns")
        .given("workers", n_workers, "agents")
        .given("p per worker", p_worker, "%", "measure it — it is not the model's benchmark score")
        .step("one agent", f"{shape.turns} turns, one transcript", single, "tok")
        .step("workers", f"{n_workers} x {split['worker_turns_each']} turns", split["worker_tokens"], "tok")
        .step("orchestrator", "briefs out, reports back", split["orchestrator_tokens"], "tok")
        .step("split total", f"{split['worker_tokens']:,} + {split['orchestrator_tokens']:,}",
              split["total"], "tok")
        .step("cost delta", f"${cost(single, model):,.2f} -> ${cost(split['total'], model):,.2f}",
              cost(split["total"], model) - cost(single, model), "$")
        .step("P(system)", f"{p_worker} {'**' if conjunctive else 'any of'} {n_workers}", p, "%")
        .check("the split is cheaper", split["total"] < single,
               "if it is not, the overhead is outrunning the quadratic — use fewer workers")
        .check(
            "fan-out did not cost you reliability",
            p >= p_worker,
            f"{p:.1%} vs {p_worker:.1%} for one agent — conjunctive fan-out multiplies failure",
        )
        .check("each agent fits comfortably", split["peak_context_per_agent"] < 200_000,
               f"peak {split['peak_context_per_agent']:,} tok per agent")
        .says(
            f"Splitting this across {n_workers} workers divides the quadratic term by {n_workers} "
            f"and gives each one its own context — but {topology.split(' ')[0]} fan-out puts system "
            f"reliability at {p:.0%}, so the parts have to be independently verifiable."
        )
    )


def when_to_fan_out(read_heavy: bool, parallelisable: bool, shared_state: bool,
                    turns: int, budget_multiplier: float = 1.0) -> dict:
    """The checklist, as a function, because the answer is usually "no".

    Fan-out pays when the work is *breadth-first over independent branches*
    and the results are small relative to what was read to produce them. It
    fails when subagents must coordinate — they cannot see each other's
    transcripts, so shared mutable state turns into two agents confidently
    doing contradictory things.
    """
    reasons = []
    verdict = True
    if not parallelisable:
        verdict, _ = False, reasons.append("the steps are sequential — a split just adds handoffs")
    if shared_state:
        verdict = False
        reasons.append("subagents write to shared state; they cannot see each other's transcripts")
    if not read_heavy:
        verdict = False
        reasons.append("results are not small relative to the reading — the reports cost as much as the work")
    if turns < 10:
        verdict = False
        reasons.append(f"{turns} turns is too short for the quadratic to matter")
    if budget_multiplier < 3:
        reasons.append("expect several times the tokens of a single agent; make sure that is authorised")
    return {
        "fan_out": verdict,
        "reasons": reasons or ["breadth-first, read-heavy, independent branches — this is the case it is for"],
    }
