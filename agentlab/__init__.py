"""agentlab — a lab kit for learning to build LLM agents by measuring them.

The modules are split by what they need, so the parts that carry the lessons
stay testable on a CPU runner with no API key and no spend:

    derive, reliability, budget, tools, trace,
    loop, sim, evals, context, security, multiagent    pure python
    providers                                          needs `anthropic` + a key
    plots                                              needs matplotlib

That split is not tidiness. An agent's loop, its context accounting, its
failure modes and its eval harness are all *your code*, and the fastest way to
never understand them is to only ever exercise them through a paid, slow,
non-deterministic API call.
"""

__version__ = "0.1.0"

from . import budget, derive, reliability, tools, trace  # noqa: F401

__all__ = ["budget", "derive", "reliability", "tools", "trace", "__version__"]


def notebook_setup(dark: bool = False, style: bool = True):
    """One call for cell 1 of a notebook: print the environment, set chart style."""
    from .env import banner

    env = banner()
    if style:
        try:
            from .plots import use_style

            use_style(dark=dark)
        except ImportError:
            print("(matplotlib not installed yet — charts will be unstyled)")
    return env
