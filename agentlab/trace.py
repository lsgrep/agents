"""Trajectories: what the agent did, in a shape you can measure.

An agent run has two outputs. One is the answer, and it is the one everybody
looks at. The other is the *trajectory* — the sequence of tool calls, the
errors, the tokens, the repeats — and that is the one that tells you whether
the answer was luck.

Two runs can both succeed with wildly different trajectories: eleven tool calls
and one, or six retries of a call that was wrong the first time and wrong the
same way five more times. Outcome-only evals score those identically. That is
how a system passes its eval set for a month while its cost per task triples.

The metric names here are deliberately shared with `agentlab.sim`, so the
dashboard you write for a simulated run plots a real one unchanged. If you can
read one chart, you can read the other.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

#: The canonical metric set. `Trace.summary()` and `sim.simulate()` both emit
#: exactly these keys — that is the point of writing them down once.
METRICS = (
    "steps",
    "tool_calls",
    "tool_errors",
    "error_rate",
    "repeat_calls",
    "distinct_calls",
    "input_tokens",
    "output_tokens",
    "context_high_water",
    "stop_reason",
    "success",
)


def _key(name: str, args: Any) -> str:
    try:
        return f"{name}({json.dumps(args, sort_keys=True, default=str)})"
    except (TypeError, ValueError):
        return f"{name}({args!r})"


@dataclass
class ToolCall:
    name: str
    input: dict = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    tokens: int = 0

    @property
    def key(self) -> str:
        """Identity for repeat detection: the tool *and* its arguments."""
        return _key(self.name, self.input)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class Turn:
    index: int
    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    context_tokens: int = 0
    stop_reason: str = "end_turn"


@dataclass
class Trace:
    """One run, start to finish."""

    task_id: str = ""
    turns: list = field(default_factory=list)
    stop_reason: str = ""
    final_text: str = ""
    success: bool | None = None
    meta: dict = field(default_factory=dict)

    # -- shape ---------------------------------------------------------------

    @property
    def steps(self) -> int:
        return len(self.turns)

    @property
    def calls(self) -> list:
        return [c for t in self.turns for c in t.tool_calls]

    @property
    def tool_calls(self) -> int:
        return len(self.calls)

    @property
    def tool_errors(self) -> int:
        return sum(1 for c in self.calls if c.is_error)

    @property
    def error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls else 0.0

    @property
    def distinct_calls(self) -> int:
        return len({c.key for c in self.calls})

    @property
    def repeat_calls(self) -> int:
        """Calls that repeat one already made with identical arguments.

        A handful is normal. A pile of them is the agent going round in a
        circle, and it is the cheapest derailment signal there is — it needs no
        judge, no labels and no model.
        """
        return self.tool_calls - self.distinct_calls

    @property
    def usage(self) -> Usage:
        total = Usage()
        for t in self.turns:
            total = total + t.usage
        return total

    @property
    def context_high_water(self) -> int:
        return max((t.context_tokens for t in self.turns), default=0)

    def longest_cycle(self, max_len: int = 4) -> int:
        """Length of the longest immediately-repeating call cycle, or 0.

        `A B A B A B` scores 2. Catching this in the harness and breaking the
        loop is worth more than any prompt telling the model not to do it.
        """
        keys = [c.key for c in self.calls]
        best = 0
        for size in range(1, max_len + 1):
            for start in range(len(keys) - 2 * size + 1):
                window = keys[start : start + size]
                reps = 1
                pos = start + size
                while keys[pos : pos + size] == window:
                    reps += 1
                    pos += size
                if reps >= 2:
                    best = max(best, size)
        return best

    def shape(self):
        """Fit a `budget.LoopShape` to this run so you can price it.

        Averages the per-turn numbers. Good enough to project what 3x the turns
        would cost; not a substitute for the real usage figures you already have.
        """
        from .budget import LoopShape

        n = max(1, self.steps)
        first = self.turns[0] if self.turns else Turn(0)
        prefix = first.context_tokens
        assistant = sum(t.usage.output_tokens for t in self.turns) // n
        results = sum(c.tokens for c in self.calls) // n
        return LoopShape(
            system_tokens=self.meta.get("system_tokens", 0),
            tool_tokens=self.meta.get("tool_tokens", 0),
            prompt_tokens=max(0, prefix - self.meta.get("system_tokens", 0) - self.meta.get("tool_tokens", 0)),
            assistant_tokens=assistant,
            result_tokens=results,
            turns=self.steps,
        )

    def summary(self) -> dict:
        u = self.usage
        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "error_rate": round(self.error_rate, 4),
            "repeat_calls": self.repeat_calls,
            "distinct_calls": self.distinct_calls,
            "input_tokens": u.input_tokens + u.cache_read_tokens + u.cache_write_tokens,
            "output_tokens": u.output_tokens,
            "context_high_water": self.context_high_water,
            "stop_reason": self.stop_reason,
            "success": self.success,
        }

    def top_tools(self, n: int = 5):
        return Counter(c.name for c in self.calls).most_common(n)

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"{self.task_id or 'run'}: {s['steps']} steps, {s['tool_calls']} calls "
            f"({s['tool_errors']} errors, {s['repeat_calls']} repeats), "
            f"{s['input_tokens']:,} in / {s['output_tokens']:,} out, "
            f"stopped on {s['stop_reason']}, success={s['success']}"
        )


def summarize(traces) -> dict:
    """Aggregate a batch. Means, plus the two tails that actually matter."""
    traces = list(traces)
    if not traces:
        return {}
    n = len(traces)
    summaries = [t.summary() for t in traces]
    done = [s for s in summaries if s["success"] is not None]
    steps = sorted(s["steps"] for s in summaries)
    return {
        "runs": n,
        "success_rate": (sum(1 for s in done if s["success"]) / len(done)) if done else None,
        "mean_steps": sum(steps) / n,
        "p90_steps": steps[min(n - 1, int(0.9 * n))],
        "mean_tool_calls": sum(s["tool_calls"] for s in summaries) / n,
        "mean_error_rate": sum(s["error_rate"] for s in summaries) / n,
        "repeat_calls": sum(s["repeat_calls"] for s in summaries),
        "input_tokens": sum(s["input_tokens"] for s in summaries),
        "output_tokens": sum(s["output_tokens"] for s in summaries),
        "peak_context": max(s["context_high_water"] for s in summaries),
        "hit_step_limit": sum(1 for s in summaries if s["stop_reason"] == "max_steps"),
    }
