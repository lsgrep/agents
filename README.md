# agents — an LLM agent engineering lab ladder

Ten labs that teach agent building by making you measure it. Each one is a Colab
notebook backed by a small tested Python package, so the notebooks stay narrative
and the logic stays honest.

**Every lab runs on a laptop.** No GPU, no API key, no spend — except lab 10,
which is the seam where you check the predictions against a real model. That is
deliberate: an agent's loop, its context accounting, its failure modes and its
eval harness are all *your code*, and the fastest way to never understand them is
to only ever exercise them through a paid, slow, non-deterministic API call.

**Where to start:** the numbering is also the reading order for the first three.
Read **01 → 02 → 03** — the loop is the thing, the loop's cost is arithmetic, and
the loop's reliability is arithmetic. After those, take the rest in any order.

The organising idea: **predict on paper, then measure, then explain the gap.**

Agent engineering looks like folklore — use ReAct, add memory, multi-agent is
better — and most of what matters is three things you can actually compute:

```python
from agentlab import budget as bg, reliability as rel

rel.derive_horizon(p_step=0.95, n_steps=20)      # 36%. Not 90%.
bg.derive_loop_tokens(shape)                     # 1.59M tokens, not 300K.
```

Every formula shows its working — givens, substitutions, sanity checks, and the
sentence you should be able to say out loud afterwards. Nobody should memorise a
cache multiplier or a context window; those are handed to you. What is being
tested is whether you can put given numbers in the right places.
[`docs/FORMULAS.md`](docs/FORMULAS.md) is the same chain as a reference sheet.

Each notebook opens with one sentence — the claim you should be able to make when
you finish it. That sentence is the deliverable; the code is how you earn the
right to say it.

## The ladder

| | Lab | Open | What you walk away with |
|---|---|---|---|
| 1 | [The agent loop, from scratch](notebooks/01_the_agent_loop.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/01_the_agent_loop.ipynb) | Twenty lines, and the four invariants that fail *silently* — including the one that quietly stops your model calling tools in parallel |
| 2 | [Context economics](notebooks/02_context_economics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/02_context_economics.ipynb) | Agent cost is **quadratic in turns**. Caching divides the quadratic by ten; only bounding the context changes its shape |
| 3 | [Reliability, and the horizon wall](notebooks/03_reliability_math.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/03_reliability_math.ipynb) | 95% per step over 20 steps is a **36% agent**. Two levers, and pass@k vs pass^k |
| 4 | [The tool surface, measured](notebooks/04_tool_surface.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/04_tool_surface.ipynb) | A CI lint for tool schemas: ambiguous pairs, unbounded output, ungated destruction — and why an error message is a prompt |
| 5 | [The doom loop](notebooks/05_the_doom_loop.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/05_the_doom_loop.ipynb) | Failure is a feedback loop with a cliff. Clearing old tool results: 72% → 100% success **and half the tokens** |
| 6 | [Evals that survive contact](notebooks/06_evals.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/06_evals.ipynb) | Trajectory vs outcome, the regression your 40-case gate cannot see, and why 92% judge agreement can be worthless |
| 7 | [Memory and context management](notebooks/07_memory_and_context.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/07_memory_and_context.ipynb) | Four strategies that fail *differently*: 100% / 30% / 70% / 100% recall, at four different prices |
| 8 | [Fan-out, and when it pays](notebooks/08_fanout.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/08_fanout.ipynb) | Fan-out divides the quadratic by `n` and multiplies your failure rate — unless the topology is the other one |
| 9 | [Security: the lethal trifecta](notebooks/09_security.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/09_security.ipynb) | Exploitability is a property of the tool surface, so it is a CI check. Every tool necessary, the vulnerability emergent |
| 10 | [Going live](notebooks/10_going_live.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/agents/blob/claude/agent-building-lessons-16749f/notebooks/10_going_live.ipynb) | The same labs against a real model: verify the token estimate, catch a silent cache invalidator, measure your own `p_step` |

Each notebook carries that badge in its own first cell too, so however you arrive
at one, it is a click away from running.

## The three results worth knowing before you open anything

**Reliability compounds.** An agent whose steps each succeed 95% of the time
completes a 20-step task 36% of the time. To hit 90% you need 99.5% per step, or
you need to get the task down to two steps. There are only ever those two levers,
and every technique in the field is one of them in disguise.

**Cost is quadratic in turns.** The transcript is resent on every request, so a
run of `n` turns bills `n*P0 + g*n(n-1)/2`. For a typical shape that is 1.59M
input tokens over 40 turns, not the 300K the prefix suggests — and 81% of the
bill is transcript you already paid for. Caching is an 80% saving and it changes
the *slope*; only bounding the context changes the *shape*.

**Failure is a feedback loop, not a decay curve.** Failures append to the
transcript, a longer transcript degrades the next step, and a degraded step fails
more. So agents hold up, hold up, hold up, and then fall over — which is why
every reliability report contains "it worked fine in testing". Short tasks never
enter the loop. In lab 5's simulator, clearing old tool results takes a 24-step
task from 72% to 100% **and halves the token bill**, with nothing about the model
changed.

## Quickstart

**Colab.** Open a notebook, run cell 1. It clones this repo and installs what
that lab needs. Every notebook is idempotent, so a disconnect costs thirty
seconds.

**Locally.**

```bash
git clone https://github.com/lsgrep/agents.git && cd agents
pip install -e ".[plot,dev]"
pytest -q                            # 175 tests, offline, ~2s

python -c "
from agentlab import reliability as rel
print(rel.derive_horizon(p_step=0.95, n_steps=20))"
```

Nothing above needs an API key. `pip install -e '.[live]'` adds the SDK for lab
10.

## What is in the package

`agentlab` is split by what each module needs, so everything that carries a
lesson stays testable on a CPU runner with no key and no spend:

| module | needs | what it is |
|---|---|---|
| `loop` | nothing | The agent loop, tools, registry, fake models, and `check_transcript` for the four invariants |
| `budget` | nothing | The cost quadratic, cache economics, bounded runs, and a price snapshot you maintain |
| `reliability` | nothing | Horizon math, retries and verifiers, pass@k / pass^k, Wilson intervals, McNemar |
| `derive` | nothing | The show-your-work renderer: givens, substitutions, sanity checks, and the sentence to say out loud |
| `tools` | nothing | Tool-surface lint (ambiguous pairs, unbounded output, ungated destruction) and error-message templates |
| `trace` | nothing | Trajectories and their metrics: repeats, cycles, error rate, context high-water |
| `sim` | nothing | The doom-loop simulator, and the lever table that moves the cliff |
| `evals` | nothing | Outcome + trajectory scoring, judge calibration, and a gate that reports its own blind spot |
| `context` | nothing | The four context strategies, measured on recall *and* tokens |
| `security` | nothing | Lethal-trifecta analysis, taint tracking, and a defensive probe corpus |
| `multiagent` | nothing | Fan-out arithmetic, topology, and cost per success |
| `plots` | matplotlib | Chart defaults: fixed hue order, one idea per axis, red reserved for failure |
| `providers` | `anthropic` | The seam: a live Claude model shaped like the fake ones, with caching placed correctly |

The simulator emits the same metric names a real `Trace` does, so lab 5's
dashboard function plots your production runs without changes. That is
deliberate: if you can read one chart you can read the other.

## Notes on the numbers

* **`agentlab.sim` is a simulator, not a benchmark.** Its absolute numbers are
  made up. The *shapes* are the lesson, and the shapes are what transfer. If you
  need a number to quote publicly, run a real benchmark.
* **`budget.PRICES` is a snapshot you maintain**, stamped with `VERIFIED_ON`.
  API pricing moves and intro rates expire; `staleness()` will warn you, but it
  cannot re-verify for you. Re-check before quoting anything to anyone.
* **`budget.estimate_tokens` is a planning heuristic** — characters over 3.8.
  JSON and schemas are denser than prose. Lab 10 §1 calibrates it against the
  real tokenizer; do that once and then plan with the estimate knowing its bias.
* **`Agent.rot` in the simulator is the parameter you must measure yourself.**
  Run your own eval at 10K, 100K and 500K tokens of filled context and fit it.
  Everyone's is different and everyone's is worse than they expect.

## Related

[`serv`](https://github.com/lsgrep/serv) is the same idea one layer down —
inference serving, KV math, GPU economics, a toy paged engine. `serv` is *under*
the model, `agents` is *around* it, and the two are complementary.

## Docs

* [`docs/FORMULAS.md`](docs/FORMULAS.md) — derive it, don't recall it. Every
  formula, where each term is read off, the trap, and the sanity check.
* [`docs/LADDER.md`](docs/LADDER.md) — why each lab exists and why it is in this
  order.
* [`docs/INTERVIEW_MAP.md`](docs/INTERVIEW_MAP.md) — every claim worth making
  mapped to the lab that produces the receipt, plus a plain list of what these
  labs do **not** cover.

## Development

```bash
pip install -e ".[dev]"
ruff check agentlab tests
pytest -q
```

CI runs both on every push, offline, in seconds. The tests cover the parts worth
trusting: the loop invariants (including the transcripts that are *wrong* in ways
that never raise), the cost and reliability arithmetic against turn-by-turn sums,
the simulator actually reproducing the cliff, the eval gate's power calculation,
and the trifecta analysis. **Every notebook's code cells are executed as part of
the suite**, because a lesson whose code does not run costs the reader their
trust and their afternoon.
