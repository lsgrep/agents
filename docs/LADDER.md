# The ladder, and why it is in this order

Each lab exists because there is a question it is the only way to answer. The
order is not arbitrary: each one uses the previous one's vocabulary.

The numbering is build order. **Read 01 → 02 → 03 first.** Those three are the
foundation: the loop is the thing, the loop's cost is arithmetic, and the
loop's reliability is arithmetic. After them the rest can be taken in any
order.

---

## 1 — The agent loop, from scratch

**Question:** what *is* an agent, exactly?

**Why first:** because "agent" is used to mean a framework, a product category
and a vibe, and none of those are things you can debug. It is twenty lines:
call the model, run the tools it asked for, put the results back, repeat.
Writing those twenty lines yourself makes every framework readable afterwards —
you can see which of the three knobs (system, tools, messages) it is turning,
and what it charges you for the privilege.

It also front-loads the four invariants, because each one fails *silently*.
Splitting parallel tool results across two user messages doesn't raise
anything; it just quietly teaches the model to stop calling tools in parallel,
and you find out from a latency chart three weeks later.

**The transferable insight:** the transcript is the state. The model is
stateless. Anything not in `messages` did not happen — and everything in
`messages` is being paid for again on every single turn. Both halves of that
sentence are the rest of the ladder.

## 2 — Context economics

**Question:** why is the bill like that?

**Why second:** it is the direct consequence of the sentence you just learned.
The transcript is resent every turn, so a run of `n` turns bills
`n*P0 + g*n(n-1)/2` — **quadratic**, not linear. Forty turns is not twice
twenty. It is about four times.

Then the three levers, in order of how much they change: caching (divides the
quadratic by ten — the biggest single win available, and it changes the slope),
bounding the context (changes the *shape*), and tool surface hygiene (reduces
`P0`, which is the term everyone optimises first because it is the visible one
and the smallest).

**The transferable insight:** cost is quadratic in turns; caching changes the
slope and only bounding the context changes the shape. And the corollary that
catches people: bounding a run that was never long enough to need it costs more
than it saves.

## 3 — Reliability, and the horizon wall

**Question:** the model is 95% accurate. Why does my agent fail half the time?

**Why third:** `0.95 ** 20 = 0.36`. That one line reframes the entire problem
from "pick a better model" to "shorten the chain or verify the steps", and it
is the reason the labs after this one exist at all.

It also introduces the honest way to report an agent: pass@k answers *can it*,
pass^k answers *can I leave it alone*, and the gap between them is where demos
live. And the uncomfortable corollary — a hundred-step task at 90% needs 99.9%
per step, which is an accuracy you cannot even establish on a 200-case eval.
Sometimes the arithmetic tells you the design is wrong before you build it,
which is the cheapest possible finding.

**The transferable insight:** there are two levers, `p_step` and `n_steps`.
Every technique in the field is one of them in disguise.

## 4 — The tool surface, measured

**Question:** why does it call the wrong tool?

**Why here:** because you now know what a tool costs (lab 2: schemas are
resent every turn) and what a wrong call costs (lab 3: it is a failed step, and
steps compound). This lab makes the surface a design artifact with measurable
properties rather than a pile of functions that accumulated.

Ambiguous pairs are the headline: `get_status` / `fetch_status` /
`query_status` is not a tool surface, it is a trick question, and the model is
doing fuzzy matching over names. `agentlab.tools.lint()` is a static check you
can run in CI over the schemas you are about to ship.

The other half is that **an error message is a prompt** — the only text in the
loop written specifically for a model that has just made a mistake and is
deciding what to do next.

**The transferable insight:** the surface is a design artifact. Fewer, sharper,
bounded tools with recoverable errors beat more tools, and you can lint for it
without a model in the loop.

## 5 — The doom loop

**Question:** what does an agent failing actually look like?

**Why here:** lab 3 gives the optimistic bound. Real agents are worse, in a
specific reproducible shape, and this is the lab where you watch it happen.

The mechanism is positive feedback and none of it is exotic: a step fails, the
failure is appended to the transcript (because the transcript is the state), a
longer transcript is a worse transcript, and a worse transcript fails more. It
holds up, holds up, holds up, and then falls over — which is why every agent
reliability report says "worked fine in testing". Short tasks never enter the
loop. **The failure is a property of the horizon, not the model.**

The payoff is the lever table, where nothing about the model changes and only
your harness does. Clearing old tool results takes a 24-step task from 72% to
100% *and halves the token bill*. The circuit breaker, honestly measured,
slightly **reduces** success — it is a cost control, not a quality lever, and
shipping it as the latter is a common and confusing mistake.

**The transferable insight:** agent failure is a feedback loop, not a decay
curve. It has a cliff, the cliff moves when you change the harness, and the
harness is yours.

## 6 — Evals that survive contact

**Question:** how would I know if it got worse?

**Why here:** everything before this produced numbers. This is where you find
out whether your measurement can support them.

Two things to grade, and they come apart: the outcome and the trajectory. An
agent that returns the right number by calling `delete_account` has not passed,
and outcome-only scoring says it has.

Then the part that makes the room quiet: a 40-case eval cannot resolve a
five-point regression. It resolves about twenty. And an LLM judge with 92%
agreement can have a kappa of 0.6 — on a set that is mostly pass, agreeing is
easy. Both are one line to check and neither is usually checked.

**The transferable insight:** an eval that cannot see the change you care about
is not conservative, it is decorative. Report the blind spot next to the
verdict.

## 7 — Memory and context management

**Question:** the run is too long. What do I drop?

**Why here:** lab 2 proved you have to bound the context. This is the lab about
what bounding costs you, which arithmetic cannot tell you.

Four strategies that fail *differently* rather than better and worse: keep
everything (perfect recall, quadratic cost, and the context that makes lab 5
fall over), clear old results (cheap, loses old detail silently and
completely), compact (keeps the gist, drops the identifiers and the exact error
text — the specifics), and handles (put a reference in the transcript, not the
payload; re-read on demand).

The last one is the one that changes the shape of the problem, and it is why
"just-in-time context" and "the filesystem is the memory" keep reappearing in
production designs. Context grows with what you *use*, not what you *saw*.

**The transferable insight:** the cheapest strategy and the one that remembers
your data are different strategies. Which you want depends on whether step 30
needs what step 4 saw — so go and look.

## 8 — Fan-out, and when it pays

**Question:** should this be several agents?

**Why here:** it needs lab 2's cost model and lab 3's reliability model at the
same time, because fan-out moves both and it moves them in opposite directions.

Splitting `m` turns across `n` subagents divides the quadratic term by `n`,
because each subagent has its own transcript. That is the real argument, and it
is about context, not speed. Then reliability: conjunctive fan-out (every
branch must work) multiplies failure — five 95% subagents is a 77% system —
while disjunctive fan-out (any branch will do) divides it. Same architecture
diagram, opposite reliability.

**The transferable insight:** minimise cost per *success*, not tokens. And say
which topology you have drawn, because that is most of the design review.

## 9 — Security: the lethal trifecta

**Question:** what is the actual attack?

**Why here:** because by now you know that the transcript is the state and that
the model cannot distinguish data from instructions inside it. The exploit
follows directly rather than being a new topic.

An agent is exploitable when it has all three of: access to private data,
exposure to untrusted content, and a way to communicate out. Any two are
survivable. All three, and anyone who can write text your agent will read — a
web page, an issue comment, a support ticket — can exfiltrate through it.

The useful part is that this is a property of your **tool surface**, so it is
static, and `security.analyze()` checks it in CI with no model and no network.
Input filtering is a speed bump; the architecture is the control.

**The transferable insight:** break a leg of the trifecta. It is the only fix
that is a fix rather than a mitigation.

## 10 — Going live

**Question:** does any of this survive a real model?

**Why last:** everything above runs offline and deterministically, deliberately
— your loop, your accounting, your failure modes and your eval harness are all
*your code*, and testing them against a paid non-deterministic API teaches you
less, slower. This lab is the seam: swap one line, run the same lessons against
Claude, and check the predictions.

The three that are worth checking against reality: the token estimate against
`count_tokens`, the cache saving against `usage.cache_read_input_tokens` (this
is the one that catches a silent invalidator), and your measured `p_step`
against the one you assumed in lab 3.

**The transferable insight:** predict on paper, then measure, then explain the
gap. Within 2x means your model of the system works. Off by 10x means a term is
missing, and finding which one is the lesson.
