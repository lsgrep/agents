"""What this machine can run, and what it cannot.

Every lab in this repo runs on a laptop with no API key. Two of them do more if
you have one. This prints which world you are in, once, so a notebook can
branch instead of failing three cells later.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Env:
    python: str
    has_anthropic_sdk: bool
    has_api_key: bool
    has_matplotlib: bool
    has_numpy: bool

    @property
    def can_run_live(self) -> bool:
        return self.has_anthropic_sdk and self.has_api_key


def detect() -> Env:
    def installed(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    return Env(
        python=".".join(str(v) for v in sys.version_info[:3]),
        has_anthropic_sdk=installed("anthropic"),
        has_api_key=bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        has_matplotlib=installed("matplotlib"),
        has_numpy=installed("numpy"),
    )


def banner() -> Env:
    env = detect()
    print(f"python {env.python}")
    print(f"matplotlib {'yes' if env.has_matplotlib else 'no — charts will not render'}")
    if env.can_run_live:
        print("anthropic sdk + key found — the live cells will run and will spend money")
    elif env.has_anthropic_sdk:
        print("anthropic sdk found, no API key — live cells will be skipped")
    else:
        print("no anthropic sdk — every lesson still runs; the live cells will be skipped")
    print("everything else here is offline, deterministic and free.")
    return env
