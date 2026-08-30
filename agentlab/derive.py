"""The show-your-work renderer.

Every piece of arithmetic in this repo is printed the same way: what you were
**given**, where each number was **read off**, the **substitution**, the answer,
and a **sanity check** that catches a unit error before it reaches a slide.

The reason for the ceremony is that agent numbers are unusually easy to get
wrong by a factor of ten and unusually hard to notice. A KV-cache mistake shows
up as an OOM. A context-cost mistake shows up as a bill, next month, once.

Pure python, no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

Number = Union[int, float]

BOX = "─"


def fmt(value, unit: str = "") -> str:
    """Format a number the way a human would read it back out loud."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if unit in ("$", "usd"):
        return f"${value:,.4f}" if abs(value) < 1 else f"${value:,.2f}"
    if unit == "%":
        return f"{value * 100:.2f}%" if abs(value) <= 1 else f"{value:.2f}%"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        text = f"{int(value):,}"
    elif abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        text = f"{value:,.4g}"
    else:
        text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


@dataclass(frozen=True)
class Given:
    """One input, and — the part that matters — where you read it off."""

    name: str
    value: Number | str
    unit: str = ""
    source: str = ""

    def render(self) -> str:
        line = f"  {self.name:<28} = {fmt(self.value, self.unit)}"
        return f"{line:<62} ({self.source})" if self.source else line


@dataclass(frozen=True)
class Step:
    label: str
    substitution: str
    value: Number | str
    unit: str = ""

    def render(self) -> str:
        return f"  {self.label:<28} = {self.substitution}\n  {'':<28} = {fmt(self.value, self.unit)}"


@dataclass(frozen=True)
class Check:
    label: str
    passed: bool
    detail: str = ""

    def render(self) -> str:
        mark = "ok  " if self.passed else "LOOK"
        return f"  [{mark}] {self.label}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class Derivation:
    """One formula, worked.

    Build it with `.given()`, `.step()`, `.check()`, then `print()` it. The
    object stays inspectable — `.value` is the answer, so a notebook can print
    the working *and* keep using the number.
    """

    title: str
    formula: str = ""
    givens: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    value: Number | None = None
    unit: str = ""
    note: str = ""

    def given(self, name, value, unit="", source="") -> Derivation:
        self.givens.append(Given(name, value, unit, source))
        return self

    def step(self, label, substitution, value, unit="") -> Derivation:
        self.steps.append(Step(label, substitution, value, unit))
        self.value, self.unit = value, unit
        return self

    def check(self, label, passed, detail="") -> Derivation:
        self.checks.append(Check(label, bool(passed), detail))
        return self

    def says(self, note: str) -> Derivation:
        """The sentence you should be able to say out loud afterwards."""
        self.note = note
        return self

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self, width: int = 74) -> str:
        out = [BOX * width, f"  {self.title}", BOX * width]
        if self.formula:
            out += [f"  {self.formula}", ""]
        if self.givens:
            out.append("  given:")
            out += [g.render() for g in self.givens]
            out.append("")
        for s in self.steps:
            out.append(s.render())
        if self.checks:
            out.append("")
            out += [c.render() for c in self.checks]
        if self.note:
            out += ["", f"  say: \"{self.note}\""]
        out.append(BOX * width)
        return "\n".join(out)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


@dataclass
class Worksheet:
    """A chain of derivations that share their givens — a whole sizing pass."""

    title: str
    parts: list = field(default_factory=list)

    def add(self, derivation: Derivation) -> Worksheet:
        self.parts.append(derivation)
        return self

    @property
    def all_checks_passed(self) -> bool:
        return all(d.all_checks_passed for d in self.parts)

    def result(self, name: str):
        """Pull one part's answer back out by title substring."""
        for d in self.parts:
            if name.lower() in d.title.lower():
                return d.value
        raise KeyError(f"no part of this worksheet is called {name!r}")

    def render(self, width: int = 74) -> str:
        head = ["=" * width, f"  {self.title}", "=" * width, ""]
        body = [d.render(width) for d in self.parts]
        tail = ["", "  every check passed." if self.all_checks_passed
                else "  SOME CHECKS WANT A SECOND LOOK — see [LOOK] above."]
        return "\n".join(head + body + tail)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()
