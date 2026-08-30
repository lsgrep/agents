"""Chart defaults, so every lab's charts read the same way.

Three rules, and they are not aesthetic:

- **One idea per axis.** A chart with cost and success on twin y-axes is two
  charts holding each other hostage.
- **Fixed hue order.** The same concept is the same colour in every lab, so you
  can compare two charts without re-reading two legends.
- **Red means bad.** It is reserved for failures, exhaustion and regressions.
  Never spend it on a neutral series.

Only this module needs matplotlib.
"""

from __future__ import annotations

#: Stable series colours. Index 0 is always the baseline / "no management" case.
SERIES = ["#3b6ea5", "#4c9f70", "#c48b3f", "#7d6ba8", "#5b8fa8", "#8a7a66"]

#: Reserved. Do not use these for ordinary series.
STATUS = {"bad": "#b4413d", "good": "#4c9f70", "warn": "#c48b3f", "muted": "#9aa0a6"}


def use_style(dark: bool = False) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "font.size": 10,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )
    if dark:
        plt.style.use("dark_background")
        plt.rcParams.update({"axes.prop_cycle": mpl.cycler(color=SERIES), "grid.alpha": 0.2})


def dashboard(rows, ax=None, title: str = "agent run"):
    """Plot a run's progress and context against step.

    Takes either `sim.Run.rows` or the per-turn records from a real `Trace`,
    because both carry the same keys. That is the point of
    `agentlab.trace.METRICS`: one chart function, two sources.
    """
    import matplotlib.pyplot as plt

    rows = list(rows)
    if ax is None:
        _, ax = plt.subplots()
    steps = [r["step"] for r in rows]
    ax.plot(steps, [r["context_tokens"] for r in rows], color=SERIES[0], label="context tokens")
    ax.set_xlabel("step")
    ax.set_ylabel("context tokens")
    ax.set_title(title)

    twin = ax.twinx()
    twin.plot(steps, [r.get("p_effective", 0) for r in rows], color=STATUS["warn"],
              linestyle="--", label="P(productive step)")
    twin.set_ylabel("P(productive step)")
    twin.set_ylim(0, 1)
    twin.grid(False)

    for row in rows:
        if row.get("outcome") in ("wrong_step", "tool_error"):
            ax.axvline(row["step"], color=STATUS["bad"], alpha=0.12)

    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    return ax
