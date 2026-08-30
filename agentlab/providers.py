"""The one module that talks to a real model.

Everything else in `agentlab` runs offline, deterministically, for free. That
is deliberate: the loop, the arithmetic, the simulator and the eval harness are
all *your code*, and testing your code against a live model is slow, expensive
and non-deterministic for no benefit.

This module is the seam. `claude()` returns something with the same call
signature as `agentlab.loop.ScriptedModel`, so every lesson in this repo runs
against the real thing by changing one line.

Requires `pip install anthropic` and an API key. Nothing else in the package
imports it, so CI never needs either.
"""

from __future__ import annotations

from .trace import Usage

DEFAULT_MODEL = "claude-opus-5"


def _client(api_key: str | None = None):
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
        raise ImportError(
            "the live labs need the SDK: pip install 'agentlab[live]'"
        ) from exc
    return Anthropic(api_key=api_key) if api_key else Anthropic()


class claude:
    """A live Claude model, shaped like the fake ones.

    Two things here are worth copying into your own harness rather than
    skimming:

    **Caching placement.** The cache is a *prefix* match: any byte that changes
    anywhere in the prefix invalidates everything after it. Render order is
    `tools` -> `system` -> `messages`, so the breakpoints go at the end of the
    stable head (tools + system, identical every turn) and at the end of the
    conversation so far. Get this wrong — a timestamp in the system prompt, a
    tool list built from an unsorted dict — and you pay full price on every
    turn while your code looks correct. `usage.cache_read_input_tokens` is the
    only proof; if it is zero across repeated calls, something is invalidating
    the prefix.

    **Thinking blocks come back in `content`.** They must be passed back
    unchanged on the next turn, which is why this returns the whole content
    list and `loop.run` appends it verbatim. Extracting just the text is the
    single most common way to break an agent loop subtly.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 8_192,
                 thinking: bool = True, effort: str | None = None,
                 cache: bool = True, api_key: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort
        self.cache = cache
        self.client = _client(api_key)
        self.responses: list = []

    def __call__(self, system, messages, tools):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._prepare(messages),
            "tools": self._tools(tools),
        }
        if system:
            block = {"type": "text", "text": system}
            if self.cache:
                # Breakpoint 1: the stable head. Same bytes on every turn of the run.
                block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [block]
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        response = self.client.messages.create(**kwargs)
        self.responses.append(response)

        from .loop import ModelResponse

        u = response.usage
        return ModelResponse(
            content=[b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in response.content],
            stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=getattr(u, "input_tokens", 0),
                output_tokens=getattr(u, "output_tokens", 0),
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
        )

    def _tools(self, tools):
        tools = [dict(t) for t in tools]
        if tools and self.cache:
            # Breakpoint 2: the end of the tool list. Tools render before the
            # system prompt, so this caches the very front of the request.
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        return tools

    def _prepare(self, messages):
        msgs = [dict(m) for m in messages]
        if msgs and self.cache:
            # Breakpoint 3: the conversation tail, so each turn reads everything
            # that came before it instead of re-ingesting it at full price.
            last = dict(msgs[-1])
            content = list(last.get("content") or [])
            if content and isinstance(content[-1], dict):
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                last["content"] = content
                msgs[-1] = last
        return msgs

    def cache_hit_rate(self) -> float | None:
        """Measured, from the usage figures. Assume nothing here.

        Returns cache reads as a share of all input tokens that *could* have
        been read. A number well below what you expect means a silent
        invalidator, and the fix is upstream in your prefix, not here.
        """
        reads = sum(getattr(r.usage, "cache_read_input_tokens", 0) or 0 for r in self.responses)
        fresh = sum(getattr(r.usage, "input_tokens", 0) for r in self.responses)
        writes = sum(getattr(r.usage, "cache_creation_input_tokens", 0) or 0 for r in self.responses)
        total = reads + fresh + writes
        return reads / total if total else None

    def spend(self) -> dict:
        """What this model instance has cost so far, from real usage numbers."""
        from .budget import MTOK, price

        p = price(self.model)
        reads = sum(getattr(r.usage, "cache_read_input_tokens", 0) or 0 for r in self.responses)
        writes = sum(getattr(r.usage, "cache_creation_input_tokens", 0) or 0 for r in self.responses)
        fresh = sum(getattr(r.usage, "input_tokens", 0) for r in self.responses)
        out = sum(getattr(r.usage, "output_tokens", 0) for r in self.responses)
        usd = (
            fresh * p.input / MTOK
            + reads * p.cache_read / MTOK
            + writes * p.cache_write() / MTOK
            + out * p.output / MTOK
        )
        return {"calls": len(self.responses), "input": fresh, "cache_read": reads,
                "cache_write": writes, "output": out, "usd": round(usd, 4)}


def count_tokens(messages, system: str = "", tools=(), model: str = DEFAULT_MODEL,
                 api_key: str | None = None) -> int:
    """The real token count, not `budget.estimate_tokens`.

    Use it once, early, to calibrate: count a representative system prompt and
    tool list for real, compare against the estimate, and then plan with the
    estimate knowing how far off it is. Guessing token counts to two significant
    figures and then arguing about the third is a waste of everyone's afternoon.
    """
    client = _client(api_key)
    kwargs = {"model": model, "messages": list(messages)}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = list(tools)
    return client.messages.count_tokens(**kwargs).input_tokens
