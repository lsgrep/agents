"""Reliability arithmetic: why a 95% agent is not a working agent.

This is the module that does the most damage to intuition, and it is four
lines of maths.

An agent that completes a task in `n` dependent steps, each succeeding
independently with probability `p`, completes the task with probability
`p ** n`. That exponent is the whole story of agent engineering. It is why
"the model is great, the demo worked" and "it fails in production" are both
true statements about the same system, and it is why almost every technique
in this repo — smaller steps, verifiers, retries, checkpoints, tighter tool
surfaces — is really an attack on one of two numbers: raise `p`, or lower `n`.

The independence assumption is generous. Real agent failures correlate: a
wrong turn early poisons every step after it, which is what `agentlab.sim`
simulates. So treat everything here as the **optimistic** bound. If the
arithmetic already says the horizon is out of reach, measurement will not
rescue it.

Also here: the estimators for reporting agent quality honestly —
`pass@k` (can it *ever*), `pass^k` (can it be *trusted*), and the confidence
interval that says whether your 40-task eval can tell 70% from 75% at all.
It usually cannot, and that is worth knowing before you ship a regression gate.

Pure python — `math` only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .derive import Derivation

# --------------------------------------------------------------------------
# The horizon
# --------------------------------------------------------------------------


def horizon_success(p_step: float, n_steps: int) -> float:
    """P(task) for `n_steps` dependent steps that each succeed with `p_step`."""
    _check_prob(p_step)
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    return p_step**n_steps


def steps_at_reliability(p_step: float, target: float) -> float:
    """How many steps you get before task success falls below `target`.

    The answer people find surprising: at 95% per step, a 90% task success rate
    buys you two steps. Not twenty.
    """
    _check_prob(p_step, strict=True)
    _check_prob(target, strict=True)
    return math.log(target) / math.log(p_step)


def required_step_accuracy(n_steps: int, target: float) -> float:
    """The per-step accuracy a horizon of `n_steps` demands to hit `target`.

    Run it on n=100 and target=0.9 and read the answer out loud. This is the
    number that explains why long-horizon agents are an engineering problem
    and not a prompting problem.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    _check_prob(target, strict=True)
    return target ** (1.0 / n_steps)


def with_retries(p_step: float, attempts: int) -> float:
    """Effective per-step success when a failed step can be detected and retried.

    `1 - (1 - p) ** attempts`. The load-bearing word is **detected**: a retry
    loop is only worth what its verifier is worth. Retrying a step whose failure
    you cannot see does nothing except spend tokens, and `p_verify` below is
    where that shows up.
    """
    _check_prob(p_step)
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    return 1.0 - (1.0 - p_step) ** attempts


def with_imperfect_verifier(p_step: float, attempts: int, p_detect: float) -> float:
    """Retries, but the verifier only catches a failure with probability `p_detect`.

    An undetected failure is not retried — it is *committed*, and every later
    step is built on it. This is the honest version of `with_retries`, and the
    reason "add a self-check step" so often buys less than expected.
    """
    _check_prob(p_step)
    _check_prob(p_detect)
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    # A step ends good if it ever succeeds; each retry only happens when the
    # previous failure was caught.
    p_fail_uncaught = (1.0 - p_step) * (1.0 - p_detect)
    p_fail_caught = (1.0 - p_step) * p_detect
    good, pending = p_step, p_fail_caught
    for _ in range(attempts - 1):
        good += pending * p_step
        pending = pending * p_fail_caught
    return good if good + p_fail_uncaught <= 1.0 + 1e-9 else min(good, 1.0)


def checkpoint_horizon(p_step: float, n_steps: int, segment: int, p_recover: float = 1.0) -> float:
    """Task success when the run is broken into recoverable segments.

    A checkpoint does not make a step more likely to succeed. It caps how much
    work a failure destroys — you re-run one segment, not the run. That turns
    one exponential of length `n` into `n / segment` exponentials of length
    `segment`, each retried once with probability `p_recover` of the retry
    landing.
    """
    if segment <= 0:
        raise ValueError("segment must be >= 1")
    _check_prob(p_recover)
    n_seg = math.ceil(n_steps / segment)
    p_seg = horizon_success(p_step, segment)
    p_seg_eff = p_seg + (1 - p_seg) * p_recover * p_seg
    return p_seg_eff**n_seg


# --------------------------------------------------------------------------
# Reporting a number honestly
# --------------------------------------------------------------------------


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k: P(at least one of k draws succeeds), from n trials.

    The HumanEval estimator. It answers a **capability** question — is the
    ability there at all, given a retry budget and someone to pick the winner.
    """
    if k > n_samples:
        raise ValueError("k must be <= n_samples")
    if n_correct > n_samples or n_correct < 0:
        raise ValueError("n_correct must be within [0, n_samples]")
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)


def pass_pow_k(p: float, k: int) -> float:
    """pass^k: P(all k independent attempts succeed) = p ** k.

    The **reliability** question, and the one production actually asks. An
    agent nobody watches has to be right every time, not once.

    The gap between the two is where agent demos live: 61% pass@1 and 25% at
    k=8 is the same agent, described honestly and dishonestly.
    """
    _check_prob(p)
    if k < 1:
        raise ValueError("k must be >= 1")
    return p**k


def pass_pow_k_empirical(trials_per_case: Sequence[Sequence[bool]], k: int) -> float:
    """pass^k measured, not assumed: the fraction of cases that pass k times.

    Takes one list of boolean outcomes per case. Uses the same combinatorial
    estimator as pass@k, so it does not need every case to have exactly k runs.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = []
    for outcomes in trials_per_case:
        n, c = len(outcomes), sum(bool(o) for o in outcomes)
        if k > n:
            raise ValueError(f"case has {n} trials, cannot estimate pass^{k}")
        if c < k:
            scores.append(0.0)
        else:
            scores.append(math.comb(c, k) / math.comb(n, k))
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------------------
# Can your eval set tell the difference?
# --------------------------------------------------------------------------

_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def _z(conf: float) -> float:
    if conf in _Z:
        return _Z[conf]
    raise ValueError(f"confidence must be one of {sorted(_Z)}")


def wilson_interval(successes: int, n: int, conf: float = 0.95):
    """Wilson score interval — the right one for small evals near 0 or 1.

    The normal-approximation interval you half-remember gives nonsense at the
    ends (17/20 succeeded, interval runs past 1.0). Wilson does not, which
    matters precisely when an agent eval is interesting.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    z = _z(conf)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def samples_for_half_width(p: float, half_width: float, conf: float = 0.95) -> int:
    """How many eval cases you need for an interval that narrow.

    Run it once with `half_width=0.05` and note the answer. Then compare it to
    the size of the eval set you were about to ship a regression gate on.
    """
    _check_prob(p)
    if not 0 < half_width < 1:
        raise ValueError("half_width must be in (0, 1)")
    z = _z(conf)
    return math.ceil(z * z * p * (1 - p) / (half_width * half_width))


def detectable_difference(n: int, p: float = 0.5, conf: float = 0.95) -> float:
    """The smallest quality difference an eval of `n` cases can even see.

    Roughly: anything smaller than this is noise you will nonetheless argue
    about in a meeting.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    z = _z(conf)
    return z * math.sqrt(2 * p * (1 - p) / n)


def mcnemar(baseline: Sequence[bool], candidate: Sequence[bool]):
    """Paired comparison of two agents on the *same* tasks.

    Returns `(n_improved, n_regressed, p_value)`. Because the same cases are
    run twice, only the disagreements carry information — the tasks both got
    right tell you nothing about which is better. This is the test a regression
    gate should be using, and an unpaired t-test on two accuracy numbers is not.

    Exact binomial two-sided p-value; no scipy.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired comparison needs the same cases in the same order")
    improved = sum(1 for b, c in zip(baseline, candidate) if not b and c)
    regressed = sum(1 for b, c in zip(baseline, candidate) if b and not c)
    n = improved + regressed
    if n == 0:
        return improved, regressed, 1.0
    lo = min(improved, regressed)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2**n)
    return improved, regressed, min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# The same arithmetic, with the working shown
# --------------------------------------------------------------------------


def derive_horizon(p_step: float, n_steps: int, target: float = 0.9) -> Derivation:
    p_task = horizon_success(p_step, n_steps)
    budget = steps_at_reliability(p_step, target)
    needed = required_step_accuracy(n_steps, target)
    return (
        Derivation("Task success over a horizon", "P(task) = p_step ** n_steps")
        .given("p_step", p_step, "%", "measured per-step success")
        .given("n_steps", n_steps, "steps", "count the tool calls in one run")
        .given("target", target, "%", "what the product needs")
        .step("P(task)", f"{p_step} ** {n_steps}", p_task, "%")
        .step("steps at target", f"log({target}) / log({p_step})", budget, "steps")
        .step("p_step needed", f"{target} ** (1/{n_steps})", needed, "%")
        .check(
            "horizon within budget",
            p_task >= target,
            f"needs p_step >= {needed:.4%}, you have {p_step:.2%}",
        )
        .check(
            "step count is the cheaper lever",
            budget < n_steps,
            f"{budget:.1f} steps of headroom vs {n_steps} spent",
        )
        .says(
            f"At {p_step:.0%} per step this run is {p_task:.1%} to finish. "
            f"To hit {target:.0%} I either get to {needed:.2%} per step or cut the run to "
            f"{budget:.0f} steps."
        )
    )


def derive_eval_power(n_cases: int, p: float = 0.7, conf: float = 0.95) -> Derivation:
    lo, hi = wilson_interval(round(p * n_cases), n_cases, conf)
    mde = detectable_difference(n_cases, p, conf)
    need = samples_for_half_width(p, 0.05, conf)
    return (
        Derivation("What your eval set can see", "half-width ≈ z * sqrt(p(1-p)/n)")
        .given("n_cases", n_cases, "cases", "the size of your eval set")
        .given("observed pass rate", p, "%", "from the run")
        .given("confidence", conf, "%")
        .step("Wilson interval", f"{round(p * n_cases)}/{n_cases}", f"[{lo:.1%}, {hi:.1%}]")
        .step("smallest visible difference", f"z * sqrt(2p(1-p)/{n_cases})", mde, "%")
        .step("cases for ±5 points", "z^2 p(1-p) / 0.05^2", need, "cases")
        .check(
            "gate can resolve a 5-point regression",
            mde <= 0.05,
            f"it currently resolves {mde:.1%}",
        )
        .says(
            f"On {n_cases} cases I can see a {mde:.0%} change, not a 2% one. "
            f"A ±5-point answer needs about {need} cases."
        )
    )


def _check_prob(p: float, strict: bool = False) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    if strict and not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
