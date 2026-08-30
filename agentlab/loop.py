"""The agent loop, from scratch.

An agent is not a framework. It is this:

    while True:
        response = model(system, messages, tools)
        if response.stop_reason != "tool_use":
            break
        messages.append(assistant(response.content))
        messages.append(user([execute(call) for call in response.tool_calls]))

Everything else — memory, planning, multi-agent, skills, guardrails — is a
choice about what goes in `system`, what goes in `tools`, and what you do to
`messages` between iterations. Once you have written the twenty lines yourself,
every framework becomes readable, because you can see which of those three
knobs it is turning and what it is charging you for the privilege.

Four invariants hold this together. Break any one of them and the failure is
quiet — a model that stops calling tools in parallel, a run that ends a turn
early, a transcript the API rejects three turns later:

1. **The transcript is the state.** The model is stateless. Anything not in
   `messages` did not happen, and anything in `messages` is being paid for
   again on every single turn.
2. **Every `tool_use` gets exactly one `tool_result`**, keyed by id. Not
   optional, not reorderable, not skippable — including for the call that
   failed.
3. **All results from one assistant turn go back in one user message.**
   Splitting parallel results across several messages silently teaches the
   model to stop making parallel calls.
4. **A tool failure is a `tool_result` with `is_error`, not an exception.**
   An exception kills the run; an error message is a chance for the model to
   recover, and is therefore a *prompt* — see `agentlab.tools`.

The model here is a protocol, and `agentlab` ships fake ones, so every lesson
in this repo runs with no API key, no network and no spend. Swap in
`agentlab.providers.claude` when you want the real thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .trace import ToolCall, Trace, Turn, Usage

# --------------------------------------------------------------------------
# Content blocks — the wire shapes, not a wrapper over them
# --------------------------------------------------------------------------


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_use_block(id: str, name: str, input: dict) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def user(content) -> dict:
    return {"role": "user", "content": [text_block(content)] if isinstance(content, str) else content}


def assistant(content) -> dict:
    return {"role": "assistant", "content": [text_block(content)] if isinstance(content, str) else content}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@dataclass
class Tool:
    """One tool: the schema the model sees, and the function the harness runs.

    `destructive` and `read_only` are not sent to the model. They are for the
    harness — what to gate behind a confirmation, what is safe to run in
    parallel, what to log loudly. That split is the entire argument for
    promoting an action out of a general-purpose `bash` tool: a typed
    `send_email(to, body)` can be gated, rendered and audited, and
    `bash("curl -X POST ...")` cannot.
    """

    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    fn: Callable[..., Any] | None = None
    read_only: bool = True
    destructive: bool = False
    tainted_output: bool = False  #: returns content from an untrusted source

    def schema(self) -> dict:
        """Exactly what goes on the wire — nothing else."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def __call__(self, **kwargs):
        if self.fn is None:
            raise NotImplementedError(f"tool {self.name!r} has no implementation")
        return self.fn(**kwargs)


class ToolRegistry:
    """The tool surface: what the model can see and what the harness will run."""

    def __init__(self, tools: Sequence[Tool] = ()):
        self._tools: dict = {}
        for t in tools:
            self.add(t)

    def add(self, tool: Tool) -> ToolRegistry:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r} — the model cannot disambiguate these")
        self._tools[tool.name] = tool
        return self

    def __contains__(self, name) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict):
        """Run one call. Returns `(content, is_error)` — it never raises.

        An unknown tool is an error *message*, not a crash: models hallucinate
        tool names, and the recoverable response is to say which names exist.
        """
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "none"
            return (f"Error: no tool named {name!r}. Available tools: {known}.", True)
        try:
            return (str(self._tools[name](**(args or {}))), False)
        except Exception as exc:  # noqa: BLE001 — deliberate: the model handles it
            return (f"Error: {type(exc).__name__}: {exc}", True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """What one call to the model returned. Mirrors the Messages API."""

    content: list
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def tool_uses(self) -> list:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


class ScriptedModel:
    """A model that replays a fixed list of responses.

    The right tool for testing the *harness*. Loop invariants, error handling,
    stop conditions and budget caps are all properties of your code, and testing
    them against a real model is slow, expensive and non-deterministic for no
    benefit.
    """

    def __init__(self, responses: Sequence[ModelResponse]):
        self.responses = list(responses)
        self.calls: list = []

    def __call__(self, system, messages, tools) -> ModelResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.responses:
            # Deliberately loud. A scripted model running dry means the harness
            # went further than the script expected, which is exactly the thing
            # you wanted to find out. Returning a plausible "end_turn" here
            # would hide it, and a silently-shortened run looks like a passing
            # test.
            raise ScriptExhausted(
                f"scripted model ran out after {len(self.calls)} calls — the loop asked for "
                "more turns than were scripted. Add responses, or lower max_steps."
            )
        return self.responses.pop(0)


class PolicyModel:
    """A model driven by a python function of the transcript.

    `policy(messages, tools) -> ModelResponse`. Enough to write an agent that
    genuinely solves a task — deterministically, offline — so a lesson can be
    about the loop rather than about the weather in the sampler.
    """

    def __init__(self, policy: Callable[[list, list], ModelResponse], tokens_per_call: int = 60):
        self.policy = policy
        self.tokens_per_call = tokens_per_call
        self.n_calls = 0

    def __call__(self, system, messages, tools) -> ModelResponse:
        self.n_calls += 1
        response = self.policy(messages, tools)
        if response.usage == Usage():
            response.usage = Usage(
                input_tokens=_transcript_tokens(system, messages, tools),
                output_tokens=self.tokens_per_call,
            )
        return response


def _transcript_tokens(system, messages, tools) -> int:
    from .budget import estimate_tokens

    total = estimate_tokens(system or "")
    total += sum(estimate_tokens(str(t)) for t in (tools or []))
    total += sum(estimate_tokens(str(m)) for m in messages)
    return total


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class ScriptExhausted(RuntimeError):
    """A `ScriptedModel` was asked for more turns than it was given."""


class LoopInvariantError(AssertionError):
    """Raised when the transcript this loop built would be rejected or misread."""


def check_transcript(messages: Sequence[dict]) -> None:
    """Assert the four invariants over a finished transcript.

    Call it in your own harness's tests. It catches the mistakes that are
    invisible at runtime and expensive later: an unanswered `tool_use`, results
    split across messages, a result with no matching call.
    """
    pending: dict = {}
    for i, msg in enumerate(messages):
        blocks = msg.get("content") or []
        if isinstance(blocks, str):
            continue
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        results = [b for b in blocks if b.get("type") == "tool_result"]
        if msg["role"] == "assistant":
            if pending:
                raise LoopInvariantError(
                    f"message {i}: assistant turn while {sorted(pending)} still awaits a tool_result"
                )
            pending = {b["id"]: b["name"] for b in uses}
        else:
            if results and msg["role"] != "user":
                raise LoopInvariantError(f"message {i}: tool_result must be sent as a user message")
            for b in results:
                tid = b.get("tool_use_id")
                if tid not in pending:
                    raise LoopInvariantError(f"message {i}: tool_result {tid!r} matches no pending tool_use")
                pending.pop(tid)
            if pending and results:
                raise LoopInvariantError(
                    f"message {i}: {sorted(pending)} left unanswered — all results from one "
                    "assistant turn must go back in one user message"
                )
    if pending:
        raise LoopInvariantError(f"transcript ends with unanswered tool_use: {sorted(pending)}")


def run(
    model,
    tools: ToolRegistry,
    prompt: str,
    system: str = "",
    max_steps: int = 20,
    max_tokens_budget: int | None = None,
    on_step=None,
    task_id: str = "",
) -> Trace:
    """Drive one agent run to completion and return its trajectory.

    Stop conditions, in the order they are checked, because *every* one of them
    fires in production and a loop with only the first is a runaway:

    - the model stops asking for tools (`stop_reason != "tool_use"`) — success;
    - `max_steps` — the circuit breaker;
    - `max_tokens_budget` — the money circuit breaker.

    `on_step(turn, messages)` runs between iterations. That hook is where
    compaction, clearing, approval gates and loop-breaking live — everything in
    this repo that "manages context" is a function you pass here.
    """
    messages: list = [user(prompt)]
    trace = Trace(task_id=task_id)
    trace.meta["system_tokens"] = _transcript_tokens(system, [], [])
    trace.meta["tool_tokens"] = sum(_transcript_tokens("", [], [s]) for s in tools.schemas())
    spent = 0

    for step in range(1, max_steps + 1):
        response = model(system, messages, tools.schemas())
        turn = Turn(
            index=step,
            text=response.text,
            usage=response.usage,
            context_tokens=_transcript_tokens(system, messages, tools.schemas()),
            stop_reason=response.stop_reason,
        )
        spent += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason != "tool_use":
            trace.turns.append(turn)
            trace.stop_reason = response.stop_reason
            trace.final_text = response.text
            break

        messages.append(assistant(response.content))

        # Invariant 3: every result from this turn goes back in ONE user message.
        results = []
        for block in response.tool_uses:
            content, is_error = tools.execute(block["name"], block.get("input") or {})
            results.append(tool_result_block(block["id"], content, is_error))
            turn.tool_calls.append(
                ToolCall(
                    name=block["name"],
                    input=block.get("input") or {},
                    result=content,
                    is_error=is_error,
                    tokens=_transcript_tokens("", [], [content]),
                )
            )
        messages.append(user(results))
        trace.turns.append(turn)

        if on_step is not None:
            on_step(turn, messages)

        if max_tokens_budget is not None and spent >= max_tokens_budget:
            trace.stop_reason = "budget"
            break
    else:
        trace.stop_reason = "max_steps"

    trace.meta["messages"] = messages
    check_transcript(messages)
    return trace
