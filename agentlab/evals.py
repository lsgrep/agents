"""Evals: the only thing standing between you and a vibe.

An agent eval has to answer a harder question than a model eval, because there
are two things to grade and they come apart:

- the **outcome** — did the task get done;
- the **trajectory** — how. Eleven tool calls or one. Three retries of the same
  wrong call. A destructive tool nobody expected it to reach for.

Outcome-only scoring says those are the same run. That is how a system holds
its pass rate for a quarter while its cost per task triples and its blast
radius quietly widens.

The other half of this module is about the eval itself, because the most
common failure in agent work is not a bad agent, it is **a gate that cannot
see**. Forty cases cannot resolve a five-point regression. An LLM judge that
agrees with you 70% of the time is a random number generator wearing a
lab coat. Both are measurable before you trust either — `judge_agreement()`
and `reliability.derive_eval_power()` do it in one line each.

Pure python; the judge is a protocol, so `calibrate` works on a heuristic
scorer, a human label set, or a real model call.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

from .reliability import mcnemar, pass_pow_k_empirical, wilson_interval
from .trace import Trace

# --------------------------------------------------------------------------
# Cases and checks
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One eval task, and everything it means to pass it.

    The trajectory fields are the ones people leave empty and later wish they
    had not. `forbids` in particular: an agent that gets the right answer by
    calling `delete_account` is not a passing run.
    """

    id: str
    prompt: str
    expect: str | None = None  #: regex the final answer must match
    requires: tuple = ()  #: tool names the run must use
    forbids: tuple = ()  #: tool names the run must never use
    max_steps: int | None = None
    tags: tuple = ()
    meta: dict = field(default_factory=dict)


@dataclass
class Score:
    case_id: str
    outcome: bool
    trajectory: bool
    reasons: tuple = ()

    @property
    def passed(self) -> bool:
        """Both halves. A run that cheats its way to the answer has not passed."""
        return self.outcome and self.trajectory


def score(case: Case, trace: Trace) -> Score:
    """Grade one run on both axes, and say why it failed."""
    reasons = []
    outcome = True
    if case.expect is not None:
        outcome = re.search(case.expect, trace.final_text or "", re.I | re.S) is not None
        if not outcome:
            reasons.append(f"answer did not match /{case.expect}/")

    used = {c.name for c in trace.calls}
    trajectory = True
    for name in case.requires:
        if name not in used:
            trajectory = False
            reasons.append(f"never called required tool {name!r}")
    for name in case.forbids:
        if name in used:
            trajectory = False
            reasons.append(f"called forbidden tool {name!r}")
    limit = case.max_steps
    if limit is not None and trace.steps > limit:
        trajectory = False
        reasons.append(f"took {trace.steps} steps, budget was {limit}")
    if trace.stop_reason in ("max_steps", "budget"):
        outcome = False
        reasons.append(f"ran out: {trace.stop_reason}")
    return Score(case.id, outcome, trajectory, tuple(reasons))


@dataclass
class Report:
    scores: list = field(default_factory=list)
    traces: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def outcome_rate(self) -> float:
        return sum(1 for s in self.scores if s.outcome) / self.n if self.n else 0.0

    def interval(self, conf: float = 0.95):
        return wilson_interval(self.passed, self.n, conf)

    def failures(self):
        return [s for s in self.scores if not s.passed]

    def __str__(self) -> str:
        lo, hi = self.interval() if self.n else (0, 0)
        gap = self.outcome_rate - self.pass_rate
        line = (
            f"{self.passed}/{self.n} passed ({self.pass_rate:.1%}, 95% CI "
            f"[{lo:.1%}, {hi:.1%}])"
        )
        if gap > 0:
            line += f"\n  {gap:.1%} of runs got the right answer the wrong way — trajectory checks caught them"
        return line


def evaluate(cases: Sequence[Case], run_fn: Callable[[Case], Trace]) -> Report:
    """Run every case through `run_fn` and grade it. `run_fn(case) -> Trace`."""
    report = Report()
    for case in cases:
        trace = run_fn(case)
        trace.task_id = trace.task_id or case.id
        s = score(case, trace)
        trace.success = s.passed
        report.scores.append(s)
        report.traces.append(trace)
    return report


# --------------------------------------------------------------------------
# Judges — calibrate before you trust
# --------------------------------------------------------------------------


def judge_agreement(judge_labels: Sequence[bool], human_labels: Sequence[bool]) -> dict:
    """Does your judge agree with you, and is the agreement better than luck?

    Raw agreement is the number people quote and it is nearly meaningless: on a
    set that is 90% pass, a judge that says "pass" unconditionally scores 90%.
    Cohen's kappa corrects for chance agreement, and it is the number to report.

    Rules of thumb: below 0.4, your judge is noise. 0.4-0.6, usable for ranking
    but not for a gate. Above 0.8, you can gate on it — and you should re-check
    it whenever the prompt, the model, or the task distribution moves.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("need one judge label per human label")
    n = len(judge_labels)
    if n == 0:
        raise ValueError("no labels to compare")
    tp = sum(1 for j, h in zip(judge_labels, human_labels) if j and h)
    tn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and not h)
    fp = sum(1 for j, h in zip(judge_labels, human_labels) if j and not h)
    fn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and h)
    agree = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    chance = p_yes + p_no
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    verdict = (
        "gate-worthy" if kappa >= 0.8 else
        "ranking only" if kappa >= 0.4 else
        "not usable — this judge is close to noise"
    )
    return {
        "n": n,
        "agreement": round(agree, 4),
        "kappa": round(kappa, 4),
        "false_pass": fp,  #: the expensive direction: bad runs waved through
        "false_fail": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@dataclass
class Gate:
    """A regression gate that reports what it can and cannot see.

    Two rules that make the difference between a gate and a ritual:

    - **Pair the runs.** The same cases, in the same order, for both versions.
      Only the cases where the two disagree carry information; an unpaired
      comparison of two pass rates throws that away and needs several times the
      data to say the same thing.
    - **Say what you could not see.** A gate that reports "no significant
      regression" on 40 cases is technically true and practically worthless.
      `blind_spot` is the size of regression this gate would have missed.
    """

    alpha: float = 0.05
    min_cases: int = 50

    def check(self, baseline: Sequence[bool], candidate: Sequence[bool]) -> dict:
        from .reliability import detectable_difference

        improved, regressed, p = mcnemar(baseline, candidate)
        n = len(baseline)
        base_rate = sum(baseline) / n if n else 0.0
        cand_rate = sum(candidate) / n if n else 0.0
        significant = p < self.alpha
        blind = detectable_difference(n, max(base_rate, 1e-6))
        return {
            "n": n,
            "baseline_rate": round(base_rate, 4),
            "candidate_rate": round(cand_rate, 4),
            "improved": improved,
            "regressed": regressed,
            "p_value": round(p, 4),
            "significant": significant,
            "verdict": (
                "REGRESSION" if significant and regressed > improved
                else "IMPROVEMENT" if significant and improved > regressed
                else "no change visible"
            ),
            "blind_spot": round(blind, 4),
            "underpowered": n < self.min_cases,
            "note": (
                f"this gate cannot see a change smaller than {blind:.1%}"
                + (f"; {n} cases is below the {self.min_cases} you set as a floor"
                   if n < self.min_cases else "")
            ),
        }


def repeat_runs(case: Case, run_fn: Callable[[Case], Trace], k: int = 5) -> list:
    """Run one case `k` times. The variance is the finding, not a nuisance."""
    return [score(case, run_fn(case)).passed for _ in range(k)]


def reliability_report(cases: Sequence[Case], run_fn: Callable[[Case], Trace], k: int = 5) -> dict:
    """pass@1 against pass^k over the same cases.

    The gap between these two numbers is the gap between a demo and a product.
    An agent nobody supervises has to be right every time.
    """
    trials = [repeat_runs(c, run_fn, k) for c in cases]
    flat = [t for case in trials for t in case]
    return {
        "cases": len(cases),
        "k": k,
        "pass@1": round(sum(flat) / len(flat), 4) if flat else 0.0,
        f"pass^{k}": round(pass_pow_k_empirical(trials, k), 4),
        "flaky_cases": sum(1 for t in trials if 0 < sum(t) < len(t)),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, conf: float = 0.95,
                 seed: int = 0):
    """Percentile bootstrap CI — for scores that are not pass/fail."""
    values = list(values)
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[math.floor((1 - conf) / 2 * n_boot)]
    hi = means[min(n_boot - 1, math.ceil((1 - (1 - conf) / 2) * n_boot) - 1)]
    return lo, hi
