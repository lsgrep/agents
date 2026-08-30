"""Execute every notebook's code cells, offline, and fail if any of them raises.

A lesson whose code does not run is worse than no lesson: it costs the reader
their trust and their afternoon. These are cheap to check because every lab in
this repo is deliberately offline and deterministic.

Two kinds of cell are skipped, both marked in the source:

    # Cell 1 — bootstrap        clones the repo and pip-installs (Colab only)
    # requires: live            needs an API key and spends money
"""

from __future__ import annotations

import json
import pathlib

import pytest

NOTEBOOKS = sorted(pathlib.Path("notebooks").glob("*.ipynb"))
SKIP_MARKERS = ("# Cell 1 — bootstrap", "# requires: live")


def cells_of(path):
    nb = json.loads(path.read_text())
    return [("".join(c["source"]), i) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"]


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS, "no notebooks found — the ladder is the point of the repo"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_runs(path):
    namespace: dict = {"__name__": "__notebook__"}
    for source, index in cells_of(path):
        if any(marker in source for marker in SKIP_MARKERS):
            continue
        try:
            exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 — we want the cell reported, not the traceback
            pytest.fail(f"{path.name} cell {index} raised {type(exc).__name__}: {exc}\n"
                        f"--- cell source ---\n{source}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_opens_with_a_colab_badge_and_a_claim(path):
    nb = json.loads(path.read_text())
    first = "".join(nb["cells"][0]["source"])
    second = "".join(nb["cells"][1]["source"])
    assert "colab-badge" in first, "however a reader arrives, it should be one click from running"
    assert "The claim you should be able to make" in second
