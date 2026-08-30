# From claim to receipt

Prep usually produces a document of things you can *say*. The risk is that
ground-level specificity is exactly what a good interviewer probes for, and a
claim you have only read collapses on the second follow-up — which, in agent
work, is always some form of *"how do you know?"*

This maps each thing worth claiming to the lab that gives you the receipt: a
number you produced yourself, on a machine you can name, with the code in
front of you.

Use it two ways: to find the lab behind a claim you feel shaky on, and — after
running a lab — to find the sentence it entitles you to say.

## The loop, and what an agent is

| The claim | The receipt | Lab |
|---|---|---|
| "An agent is a loop over model, tools, transcript — I've written the twenty lines" | Your own loop, running, with the four invariants asserted in a test | [01](../notebooks/01_the_agent_loop.ipynb) |
| "The transcript is the state, and that's why it's expensive" | The same fact deriving both the cache design and the cost quadratic | [01](../notebooks/01_the_agent_loop.ipynb), [02](../notebooks/02_context_economics.ipynb) |
| "Parallel tool results must go back in one user message" | `check_transcript` failing on the split version, before a model ever sees it | [01](../notebooks/01_the_agent_loop.ipynb) |
| "A tool failure is a `tool_result`, not an exception — it's a recovery opportunity" | The run that survives a `ZeroDivisionError` and finishes | [01](../notebooks/01_the_agent_loop.ipynb) |
| "Stop conditions are three, and all three fire in production" | Step limit, token budget and completion, each triggered on purpose | [01](../notebooks/01_the_agent_loop.ipynb) |

## Cost, and the conversation with finance

| The claim | The receipt | Lab |
|---|---|---|
| "Agent cost is quadratic in turns — 40 turns isn't twice 20, it's four times" | The closed form, checked against a turn-by-turn sum | [02](../notebooks/02_context_economics.ipynb) |
| "Caching divides the quadratic by ten; only bounding the context changes its shape" | Both curves, plotted, with the doubling ratio at 20/40/80/160 turns | [02](../notebooks/02_context_economics.ipynb) |
| "Our cache hit rate is measured, not assumed" | `usage.cache_read_input_tokens` against what you expected to be read | [02](../notebooks/02_context_economics.ipynb), [10](../notebooks/10_going_live.ipynb) |
| "Compaction on a short run costs more than it saves — here's the crossover" | The turn count where bounding starts paying, for your shape | [02](../notebooks/02_context_economics.ipynb) |
| "Thirty tools is 11K tokens of every single request" | Your surface priced, as a share of the window | [04](../notebooks/04_tool_surface.ipynb) |
| "Total tokens is the bill; peak context is the wall — different numbers" | Both, for the same run | [02](../notebooks/02_context_economics.ipynb) |

## Reliability, and why the demo worked

| The claim | The receipt | Lab |
|---|---|---|
| "95% per step over 20 steps is a 36% agent" | The horizon derivation, on your own measured `p_step` | [03](../notebooks/03_reliability_math.ipynb) |
| "A 100-step task at 90% needs 99.9% per step — which I couldn't even measure" | The required-accuracy line next to the eval-power line | [03](../notebooks/03_reliability_math.ipynb), [06](../notebooks/06_evals.ipynb) |
| "There are two levers: raise p_step or lower n_steps" | The sweep where each one moves the curve | [03](../notebooks/03_reliability_math.ipynb) |
| "A retry loop is worth exactly what its verifier is worth" | Perfect vs imperfect verifier, same retry budget, different horizon | [03](../notebooks/03_reliability_math.ipynb) |
| "pass@k is a capability claim; pass^k is the one production asks" | Both computed on the same runs, with the flaky cases named | [03](../notebooks/03_reliability_math.ipynb), [06](../notebooks/06_evals.ipynb) |
| "Checkpoints don't raise p_step — they cap what a failure destroys" | Segmented vs unsegmented horizon at the same step accuracy | [03](../notebooks/03_reliability_math.ipynb) |

## Debugging a failing agent

| The claim | The receipt | Lab |
|---|---|---|
| "Agent failure is a feedback loop, not a decay curve — I've watched the cliff" | The horizon sweep crossing the independent bound, once, in one direction | [05](../notebooks/05_the_doom_loop.ipynb) |
| "Clearing old tool results took us from 72% to 100% and halved the bill" | The lever table, with the model held constant | [05](../notebooks/05_the_doom_loop.ipynb) |
| "A circuit breaker is a cost control, not a quality lever" | The row where it *lowers* success and lowers spend | [05](../notebooks/05_the_doom_loop.ipynb) |
| "Repeat calls with identical arguments are the cheapest derailment signal there is" | `repeat_calls` and `longest_cycle` on a real trajectory — no judge, no labels | [01](../notebooks/01_the_agent_loop.ipynb), [05](../notebooks/05_the_doom_loop.ipynb) |
| "It calls the wrong tool because two of them are indistinguishable" | `lint()` flagging the pair before a model ever saw them | [04](../notebooks/04_tool_surface.ipynb) |
| "Context rot is a feedback term, not a footnote" | P(productive step) plotted against context fill, with the failures marked | [05](../notebooks/05_the_doom_loop.ipynb) |

## Evals, and gates that mean something

| The claim | The receipt | Lab |
|---|---|---|
| "We grade the trajectory, not just the outcome" | The agent that got the right answer by calling the forbidden tool, failing | [06](../notebooks/06_evals.ipynb) |
| "A 40-case eval can't see a 5-point regression — it sees about 20" | The blind spot printed next to the verdict | [06](../notebooks/06_evals.ipynb) |
| "Our judge has a kappa, not just an agreement number" | 92% agreement, kappa 0.6, verdict 'ranking only' | [06](../notebooks/06_evals.ipynb) |
| "We compare paired, with McNemar — only the disagreements carry information" | The gate, and the unpaired version needing several times the data | [06](../notebooks/06_evals.ipynb) |
| "A judge's false passes cost more than its false fails" | Both reported separately, because they are not the same mistake | [06](../notebooks/06_evals.ipynb) |

## Context and memory

| The claim | The receipt | Lab |
|---|---|---|
| "The four strategies fail differently, not better and worse" | Recall against tokens for all four on the same task | [07](../notebooks/07_memory_and_context.ipynb) |
| "Compaction keeps the gist and drops the identifiers" | 70% recall, and which 30% went | [07](../notebooks/07_memory_and_context.ipynb) |
| "Handles beat everything: full recall at the lowest token cost" | ...and the `+10 extra calls` that is the actual price | [07](../notebooks/07_memory_and_context.ipynb) |
| "A compaction threshold above your runs' peak never fires" | The strategy that silently did nothing at all | [07](../notebooks/07_memory_and_context.ipynb) |

## Multi-agent

| The claim | The receipt | Lab |
|---|---|---|
| "Fan-out is a context play, not a speed play — it divides the quadratic by n" | The algebra, and the token curve | [08](../notebooks/08_fanout.ipynb) |
| "Five 95% subagents that all have to work is a 77% system" | Conjunctive against disjunctive on the same diagram | [08](../notebooks/08_fanout.ipynb) |
| "We optimise cost per success, not cost — otherwise the answer is always 'more workers'" | The two curves, and the optimum only one of them has | [08](../notebooks/08_fanout.ipynb) |
| "Subagents can't see each other's transcripts, so shared mutable state is the failure mode" | The checklist, and the cases it says no to | [08](../notebooks/08_fanout.ipynb) |

## Security

| The claim | The receipt | Lab |
|---|---|---|
| "The lethal trifecta is a property of the tool surface, so we check it in CI" | `analyze()` on your own surface, in a test that fails | [09](../notebooks/09_security.ipynb) |
| "We were one convenience tool away from critical" | The two-of-three finding, naming the missing leg | [09](../notebooks/09_security.ipynb) |
| "Input filtering is a speed bump; breaking a leg is the control" | The probe corpus, and what still gets through a filter that blocks all six | [09](../notebooks/09_security.ipynb) |
| "We can see when an outbound call followed untrusted input" | Taint events on a real trajectory | [09](../notebooks/09_security.ipynb) |

---

## What these labs do not cover

Being straight about this is worth more than another table row, and an
interviewer who knows the field will respect the boundary more than a bluff.

- **Training, fine-tuning and RL on agent trajectories.** Everything here treats
  the model as fixed and optimises the harness around it. That is the right
  default for almost everyone, and it is not the whole field.
- **Real benchmark numbers.** No SWE-bench, no τ-bench, no Terminal-Bench runs.
  The simulator's absolute numbers are *made up*; only the shapes transfer. If
  you need a number to quote publicly, run the benchmark.
- **Serving and inference performance.** Latency, batching, KV cache, GPU
  economics — that is [`serv`](https://github.com/lsgrep/serv)'s ladder, and
  the two are complementary: `serv` is under the model, `agents` is around it.
- **Retrieval quality.** Chunking, embeddings, rerankers and recall@k are
  treated here only as a context-management strategy. `serv`'s lab 8 does
  retrieval properly.
- **Prompt engineering as a craft.** Almost absent, deliberately. It matters,
  it is well covered elsewhere, and it is not where agent runs actually fail.
- **Human factors.** Approval UX, escalation design, and how people actually
  supervise a fleet of agents. Mentioned in lab 9, not measured anywhere.
- **Multi-turn user conversation.** These are task-completion agents. A chat
  product has different failure modes and a different cost profile.
