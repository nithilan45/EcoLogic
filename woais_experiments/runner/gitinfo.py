"""Read-only git snapshot for a run. Never mutates git config or history."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from woais_experiments.paths import ROOT


def _git(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return 127, str(exc)
    out = (proc.stdout or proc.stderr or "").strip()
    return int(proc.returncode), out


def git_snapshot() -> dict[str, Any]:
    code, commit = _git("rev-parse", "HEAD")
    if code != 0:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": commit or "git rev-parse failed",
        }
    _, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty_code, dirty_out = _git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": branch or None,
        "dirty": dirty_code == 0 and bool(dirty_out),
        "error": None,
    }
