# Derive it, don't recall it

Nobody should walk into a conversation having memorised that Claude Opus 5 has
a 1M context window or that a cache read bills at 0.1x. Those numbers are
**given** to you — on a pricing page, in a config, by the person asking. What
is being tested is whether you can put them in the right places, in the right
order, and say what the answer means.

So this is not a numbers sheet. It is the set of formulas, what each term is,
**where you read it off**, the trap, and the sanity check that catches a unit
error before it reaches a slide.

Every formula here has a function in `agentlab` that takes raw scalars and
prints the substitution. Use those to check yourself, never to produce an
answer you did not work out first.

```python
from agentlab import budget as bg, reliability as rel, multiagent as ma

rel.derive_horizon(p_step=0.95, n_steps=20)
bg.worksheet(bg.LoopShape(system_tokens=1200, tool_tokens=6000, prompt_tokens=300,
                          assistant_tokens=250, result_tokens=1400, turns=40))
ma.derive_fanout(shape, n_workers=4, p_worker=0.95)
```

---

## 1. The horizon — the one that matters most

```
P(task) = p_step ** n_steps
```

| Term | Read it from | Trap |
|---|---|---|
| `p_step` | your own traces: productive steps / total steps | **Not the model's benchmark score.** That is measured on single turns with clean inputs. Your agent's per-step rate includes your tools, your schemas and your error messages |
| `n_steps` | count the tool calls in one real run | People quote the *happy path*. Count a median run, including the retries |

**Sanity check:** at 95% per step, twenty steps is 36%. If your intuition said
"about 90%", that gap is the reason this page exists.

**Say out loud:** *"Per-step accuracy compounds. The question isn't how good
the model is, it's how many dependent steps I'm asking it to chain."*

## 2. What a horizon demands

```
steps_at_target      = log(target) / log(p_step)
p_step needed        = target ** (1 / n_steps)
```

At `p_step = 0.95`, a 90% task success rate buys you **two steps**. A hundred
step task at 90% needs **99.9% per step** — an accuracy you cannot even
*establish* on a 200-case eval (see §7).

**Say out loud:** *"There are only two levers: raise p_step, or lower n_steps.
Everything else is one of those two wearing a costume."*

## 3. Retries, and what a verifier is worth

```
with a perfect verifier:    p_eff = 1 - (1 - p_step) ** attempts
with an imperfect one:      failures you don't detect are committed, not retried
```

| Term | Read it from | Trap |
|---|---|---|
| `attempts` | your retry policy | Retries cost turns, and turns resend the transcript — a retry is not cheap |
| `p_detect` | measure it: inject known-bad steps, count how many your check catches | **This is the load-bearing term.** A retry loop is worth exactly what its verifier is worth. Retrying a failure you cannot see does nothing but spend tokens |

**Say out loud:** *"Adding a self-check step buys you `p_detect`, not
certainty — and an undetected error isn't retried, it's built on."*

## 4. What a run costs

```
total_input = n * P0 + g * n(n-1)/2

P0 = system + tool schemas + prompt      (resent, in full, every turn)
g  = assistant tokens + tool result tokens  (how much longer the transcript gets each turn)
```

| Term | Read it from | Trap |
|---|---|---|
| `P0` | `count_tokens` on your system prompt + tool list | Tool schemas are the part people forget. 30 tools ≈ 11K tokens, **on every request** |
| `g` | a real trace. `Trace.shape()` fits it | `result_tokens` is almost always bigger than you think. A tool that returns a file returns it into a transcript that is resent forever |
| `n` | turns in a median run | The quadratic means the *tail* of your run-length distribution dominates your bill |

**Sanity check:** the growing term should dominate the fixed one past ~10
turns. If it doesn't, your tool results are suspiciously small — check you are
measuring a real run and not a demo.

**Say out loud:** *"Agent cost is quadratic in turns, not linear. Forty turns
isn't twice twenty, it's about four times."*

## 5. Peak context — a different question from cost

```
peak_context = P0 + g * (n - 1)
```

Cost is the **sum** of every request. Fitting in the window is about the
**largest** one. They are different numbers and they fail differently: one
arrives as a bill, the other as a hard error mid-run.

**Say out loud:** *"Total tokens is the bill; peak context is the wall."*

## 6. Caching

```
cache read  = 0.10x input          cache write (5m) = 1.25x input
```

Turn `i` reads the transcript as of turn `i-1` and writes only the delta.

| Term | Read it from | Trap |
|---|---|---|
| multipliers | the pricing page. `budget.CACHE_READ` etc. are a dated snapshot | Re-check them; `budget.staleness()` will tell you how old the snapshot is, it cannot re-verify it |
| hit rate | `usage.cache_read_input_tokens`, measured | **Assume nothing here.** A timestamp in your system prompt, an unsorted tool list or a 5-minute stall silently costs you the entire saving while the code looks correct |

**What caching does and does not do:** token counts are unchanged; the
quadratic term moves onto the read rate. Typically an 80%+ saving. The dollar
curve is *flattened at moderate run lengths* — the write and output terms are
linear — and **steepens again as runs grow**, converging back toward the
uncached doubling ratio. It is the highest-leverage single change you can
make, and it does not change the shape.

**Sanity check:** if `cache_read_input_tokens` is 0 across repeated calls, you
are not caching. Find the invalidator before you tune anything else.

**Say out loud:** *"Caching divides the quadratic by ten. Only bounding the
context changes its shape."*

## 7. Can your eval see anything?

```
half-width       ≈ z * sqrt(p(1-p)/n)
smallest visible ≈ z * sqrt(2p(1-p)/n)
n for ±5 points  ≈ z² p(1-p) / 0.05²
```

At n=40 and p=0.7, the smallest difference you can resolve is about **21
points**. A gate on 40 cases reporting "no regression" is reporting that it
looked.

Use Wilson, not the normal approximation — at 20/20 the textbook interval runs
past 1.0, and 20/20 is exactly the case an agent eval produces.

Compare two versions **paired**, on the same cases, with McNemar: only the
cases where they disagree carry information.

**Say out loud:** *"Before I trust this gate, what size regression could it
have missed?"*

## 8. pass@k and pass^k

```
pass@k = 1 - C(n-c, k) / C(n, k)      "is the ability there at all"
pass^k = p ** k                        "can it be trusted unsupervised"
```

Same agent, two numbers, and the gap between them is the gap between a demo
and a product. 61% at pass@1 and 25% at pass^8 is one honest description and
one dishonest one of the same system.

**Say out loud:** *"pass@k is a capability claim. pass^k is a reliability
claim. Production asks the second one."*

## 9. Fan-out

```
one agent:   m*P0 + g*m²/2
n subagents: m*P0 + g*m²/(2n)  + briefs + reports
```

Each subagent has **its own transcript**, so the quadratic term is divided by
`n`. That is the real argument for multi-agent, and it is a context argument,
not a speed one.

Reliability goes the other way, and the topology decides which way:

```
conjunctive (all must succeed):  p ** n        — fan-out multiplies failure
disjunctive (any will do):       1-(1-p) ** n  — fan-out divides it
```

Five 95% subagents that all have to succeed is a **77%** system.

**Sanity check:** minimise **cost per success** (`usd / P(success)`), not
tokens. Token cost alone falls monotonically in `n` and will always tell you to
add more workers.

**Say out loud:** *"Which topology is this — does every branch have to work,
or does any one of them?"*

## 10. Tool surface overhead

```
per-request overhead = n_tools * tokens_per_schema
run overhead         = that, times every turn
```

A 150-tool MCP surface at ~380 tokens a schema is ~57K tokens of every single
request before the agent does anything. Selection accuracy also degrades, and
it degrades fastest between tools that *look alike* — the model is doing fuzzy
matching over names and descriptions.

**Say out loud:** *"What is my tool surface costing me per turn, and how many
of those tools were relevant to this request?"*

---

## The two sanity checks worth doing every time

1. **Multiply it out to a run.** A number per turn is not a number. Twelve
   hundred tokens per tool result, forty turns, quadratic — that is 1.6M input
   tokens, and now you can tell whether you care.
2. **Ask what would have to be true.** If the arithmetic says you need 99.9%
   per step, ask how you would *measure* 99.9% before asking how to achieve it.
   Usually you cannot, and that tells you the design is wrong rather than the
   model.
