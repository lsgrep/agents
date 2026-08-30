"""Designing a tool surface, and measuring what it costs you.

Two facts sit behind almost every "the model is being stupid" report:

1. **Tool schemas are resent on every request.** Thirty tools at ~380 tokens
   each is ~11K tokens of overhead per turn, before the agent does anything.
   Over a forty-turn run that is 450K input tokens spent describing tools that
   were mostly irrelevant.
2. **Selection accuracy falls as the surface grows**, and it falls fastest
   among tools that *look alike*. `get_status`, `fetch_status` and
   `query_status` in one namespace is not a tool surface, it is a trick
   question. The model is doing fuzzy matching over names and descriptions,
   and three near-identical strings is exactly the case fuzzy matching gets
   wrong.

So the tool surface is a design artifact with measurable properties, and this
module measures them. `lint()` is a static check you can run in CI over the
schemas you are about to ship — no model, no API key, no spend.

The other half of tool design is that **an error message is a prompt.** It is
the only text in the loop written specifically for a model that has just made a
mistake and is about to decide what to do next. `error()` is a template for
writing them like it matters: what failed, why, and what to do instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .budget import context_share, estimate_tokens

#: Verbs people reach for interchangeably. Two tools whose names differ only by
#: one of these, over the same noun, are a coin flip at selection time.
SYNONYM_VERBS = (
    {"get", "fetch", "read", "retrieve", "load", "query", "lookup", "find"},
    {"list", "search", "browse", "enumerate", "scan"},
    {"create", "add", "new", "insert", "make", "post"},
    {"update", "edit", "modify", "patch", "change", "set"},
    {"delete", "remove", "drop", "destroy", "purge"},
    {"run", "exec", "execute", "invoke", "call"},
)

_WORD = re.compile(r"[a-z0-9]+")

#: Words that appear in every tool description ever written. Leaving them in
#: makes any two descriptions look 60% alike and buries the real collisions.
_STOP = frozenset(
    "a an the of for from in on to and or with by this that it its is are be "
    "return returns returning use used using call calls given get gets set "
    "record records system service api tool value values data id".split()
)
_LIST_HINT = re.compile(r"\b(list|search|all|every|entries|results|records|rows|files|items)\b")
_BOUND_PARAM = re.compile(r"limit|max|top_?k|page|per_page|count|cursor|offset|n_results")


def _words(text: str) -> list:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


def _split_name(name: str):
    parts = _WORD.findall(name.lower())
    return (parts[0], tuple(parts[1:])) if parts else ("", ())


def _same_verb_family(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in family and b in family for family in SYNONYM_VERBS)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


# --------------------------------------------------------------------------
# What the surface costs
# --------------------------------------------------------------------------


def schema_tokens(schema: dict) -> int:
    """Token cost of one tool definition as sent on the wire.

    Estimated (see `budget.CHARS_PER_TOKEN`). For a number you can put in a
    budget, count the real thing — `providers.count_tokens` sends the tool list
    to the token-counting endpoint.
    """
    return estimate_tokens(json.dumps(schema, separators=(",", ":")))


def surface_tokens(tools: Sequence) -> int:
    return sum(schema_tokens(_schema(t)) for t in tools)


def _schema(tool) -> dict:
    return tool.schema() if hasattr(tool, "schema") else dict(tool)


# --------------------------------------------------------------------------
# The lint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str  #: "high" | "medium" | "low"
    tools: tuple
    message: str
    fix: str = ""

    def __str__(self) -> str:
        who = ", ".join(self.tools) if self.tools else "surface"
        line = f"[{self.severity:<6}] {self.kind:<18} {who}: {self.message}"
        return f"{line}\n{'':<29}fix: {self.fix}" if self.fix else line


def lint(tools: Sequence, model: str = "claude-opus-5", budget_share: float = 0.05) -> list:
    """Static checks over a tool surface. Run it in CI.

    None of these are style points. Each one corresponds to a failure you will
    otherwise diagnose from a confused transcript at 11pm:

    - **ambiguous_pair** — two tools a fuzzy match cannot separate.
    - **duplicate_name** — the registry will reject it; the model would too.
    - **surface_cost** — the schemas alone eat a chunk of every request.
    - **thin_description** — the description is the entire selection signal.
    - **unbounded_output** — a tool that can return an unbounded blob will one
      day return one, into a transcript that is resent every turn afterwards.
    - **untyped_choice** — a free-text parameter where an enum was meant.
    - **ungated_destructive** — a hard-to-reverse action with no confirmation
      hook. The harness cannot gate what it cannot name.
    """
    findings: list = []
    schemas = [_schema(t) for t in tools]
    names = [s["name"] for s in schemas]

    seen = set()
    for name in names:
        if name in seen:
            findings.append(
                Finding("duplicate_name", "high", (name,), "two tools share this name", "namespace them")
            )
        seen.add(name)

    total = surface_tokens(tools)
    share = context_share(total, model)
    if share > budget_share:
        findings.append(
            Finding(
                "surface_cost",
                "medium" if share < 2 * budget_share else "high",
                (),
                f"{len(schemas)} tools = ~{total:,} tokens, {share:.1%} of the window, on every request",
                "defer loading with tool search, or split the surface across sub-agents",
            )
        )

    for i, a in enumerate(schemas):
        for b in schemas[i + 1 :]:
            va, na = _split_name(a["name"])
            vb, nb = _split_name(b["name"])
            desc = _jaccard(_words(a.get("description", "")), _words(b.get("description", "")))
            if na and na == nb and _same_verb_family(va, vb):
                findings.append(
                    Finding(
                        "ambiguous_pair",
                        "high",
                        (a["name"], b["name"]),
                        "same object, interchangeable verb — selection is close to a coin flip",
                        "merge them, or make each description say when NOT to use it",
                    )
                )
            elif desc >= 0.6 and na and na == nb:
                # Similarity alone is not a finding — every tool description in
                # a codebase shares boilerplate, and two tools over *different*
                # objects are told apart by the object. It only matters when
                # the names name the same thing, because then the description
                # is the only thing left to separate them, and it isn't.
                findings.append(
                    Finding(
                        "ambiguous_pair",
                        "medium",
                        (a["name"], b["name"]),
                        f"related names and {desc:.0%} the same words in the description",
                        "state the distinguishing case in the first sentence of each",
                    )
                )

    for tool, schema in zip(tools, schemas):
        name, desc = schema["name"], schema.get("description", "")
        props = (schema.get("input_schema") or {}).get("properties") or {}
        if len(desc.split()) < 6:
            findings.append(
                Finding(
                    "thin_description",
                    "medium",
                    (name,),
                    f"{len(desc.split())} words — this is the model's only selection signal",
                    "say what it returns, when to use it, and when to use something else",
                )
            )
        if _LIST_HINT.search(f"{name} {desc}") and not any(_BOUND_PARAM.search(p) for p in props):
            findings.append(
                Finding(
                    "unbounded_output",
                    "medium",
                    (name,),
                    "returns a collection with no limit/cursor parameter",
                    "add `limit` with a sane default; a 40K-token result is resent every later turn",
                )
            )
        for pname, prop in props.items():
            if prop.get("type") == "string" and not prop.get("enum") and re.search(
                r"^(type|kind|mode|format|status|level|sort|order|direction)$", pname
            ):
                findings.append(
                    Finding(
                        "untyped_choice",
                        "low",
                        (name,),
                        f"parameter {pname!r} is free text where a fixed set was meant",
                        "add an enum — invalid values become a wasted turn",
                    )
                )
        if getattr(tool, "destructive", False) and getattr(tool, "read_only", False):
            findings.append(
                Finding("ungated_destructive", "high", (name,), "marked destructive and read_only", "pick one")
            )

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: (order[f.severity], f.kind, f.tools))


def report(tools: Sequence, model: str = "claude-opus-5", group_after: int = 3) -> str:
    """A readable summary. `lint()` is the full list; this is the thing you read.

    Findings of the same kind are collapsed past `group_after`, because a
    surface with one bad pattern repeated forty times has one problem, not
    forty, and a wall of identical lines hides the other findings.
    """
    findings = lint(tools, model)
    total = surface_tokens(tools)
    out = [
        f"{len(list(tools))} tools, ~{total:,} tokens per request "
        f"({context_share(total, model):.2%} of the {model} window)",
        "",
    ]
    if not findings:
        return "\n".join(out + ["  no findings."])

    by_kind: dict = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)
    for kind, group in by_kind.items():
        for f in group[:group_after]:
            out.append(str(f))
        if len(group) > group_after:
            rest = sorted({t for f in group[group_after:] for t in f.tools})
            out.append(f"{'':<9}… and {len(group) - group_after} more {kind}: {', '.join(rest[:12])}"
                       + (" …" if len(rest) > 12 else ""))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Errors are prompts
# --------------------------------------------------------------------------


def error(what: str, why: str = "", do_next: str = "", valid: Sequence[str] | None = None) -> str:
    """Compose a tool error the model can actually recover from.

    Compare the two versions of the same failure:

        "Error: 400"

        "Error: search_orders failed — `status` must be one of open, shipped,
         cancelled (got 'pending'). Retry with a valid status, or call
         list_order_statuses first."

    The second one costs you thirty tokens and saves a turn, and turns are the
    expensive unit: each one resends the whole transcript.
    """
    parts = [f"Error: {what}"]
    if why:
        parts.append(why.rstrip("."))
    if valid:
        parts.append("Valid values: " + ", ".join(str(v) for v in valid))
    if do_next:
        parts.append(do_next.rstrip(".") + ".")
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


# --------------------------------------------------------------------------
# Material for the selection experiment
# --------------------------------------------------------------------------

_NOUNS = ("order", "invoice", "ticket", "customer", "shipment", "refund", "account",
          "subscription", "payment", "review", "inventory", "coupon", "contract",
          "vendor", "warehouse", "returns_case", "dispute", "credit_note", "quote",
          "campaign", "segment", "webhook_log", "audit_entry", "tax_rate",
          "price_list", "carrier", "batch", "sku")
_VERBS = ("get", "list", "search", "update", "cancel", "create")
_PHRASING = {
    "get": "Look up a single {noun} by its identifier and return its full detail.",
    "list": "Page through {noun}s belonging to one account, newest first.",
    "search": "Free-text search across {noun}s; matches title and body fields.",
    "update": "Apply a partial change to an existing {noun}. Fields omitted are left alone.",
    "cancel": "Void a {noun}. This cannot be undone once the downstream job has run.",
    "create": "Open a new {noun} and return the identifier assigned to it.",
}


def _distractor_schema(verb: str, noun: str) -> dict:
    if verb in ("list", "search"):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": f"Text to match against {noun}s."},
                "limit": {"type": "integer", "description": "Maximum rows to return.", "default": 20},
            },
            "required": ["query"],
        }
    return {
        "type": "object",
        "properties": {f"{noun}_id": {"type": "string", "description": f"The {noun} id."}},
        "required": [f"{noun}_id"],
    }


def distractors(n: int, seed: int = 0, collisions: bool = True) -> list:
    """Plausible-but-irrelevant tools, for measuring what surface size costs.

    Deliberately *realistic* rather than random: a surface degrades because its
    tools resemble each other, so distractors that read like a real back-office
    API are the honest test. Random noise tools are an easy test that passes.

    Because it is realistic, a large generated surface grows collisions on its
    own — which is the finding, not a defect. Pass `collisions=False` for a
    deliberately clean control surface, so an experiment can vary surface *size*
    without also varying surface *quality*.
    """
    import random

    from .loop import Tool

    if n < 0:
        raise ValueError("n must be >= 0")
    capacity = len(_VERBS) * len(_NOUNS) if collisions else len(_NOUNS)
    if n > capacity:
        # Better a clear error than a loop that spins forever looking for a
        # unique name that cannot exist.
        raise ValueError(
            f"can only generate {capacity} distinct tools from this vocabulary, asked for {n}"
        )
    rng = random.Random(seed)
    out, used, claimed_nouns = [], set(), set()
    while len(out) < n:
        verb, noun = rng.choice(_VERBS), rng.choice(_NOUNS)
        name = f"{verb}_{noun}"
        if name in used or (not collisions and noun in claimed_nouns):
            continue
        used.add(name)
        claimed_nouns.add(noun)
        out.append(
            Tool(
                name=name,
                description=_PHRASING[verb].format(noun=noun),
                input_schema=_distractor_schema(verb, noun),
                read_only=verb in ("get", "list", "search"),
                destructive=verb == "cancel",
            )
        )
    return out
