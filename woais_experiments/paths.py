"""Repo paths and a one-way door: frozen trees are read-only from this package."""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
RESULTS = PACKAGE / "results"
RUNS = RESULTS / "runs"
CONFIGS = PACKAGE / "configs"
HASH_MANIFEST = PACKAGE / "EXISTING_RESULTS_SHA256.txt"


_HOME_ABS_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/folders/",
    "/private/var/folders/",
    "/tmp/",
    "/private/tmp/",
)


def repo_rel(path: Path | str) -> str:
    """Path relative to the repo root when possible (no machine-absolute strings)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def looks_like_home_absolute(text: str) -> bool:
    """True for machine-home / temp paths that would deanonymize an author."""
    n = text.replace("\\", "/")
    if n.startswith(_HOME_ABS_PREFIXES):
        return True
    if len(text) >= 3 and text[1] == ":" and ("/Users/" in n or "/users/" in n.lower()):
        return True
    return False


def public_relpath(path: Path | str) -> str:
    """Repo-relative path for JSON artifacts. Home/temp absolutes become a token."""
    rel = repo_rel(path)
    if looks_like_home_absolute(rel):
        return "<redacted-absolute>"
    return rel.replace("\\", "/")

# replace: current run_offline / test behavior (atomic overwrite).
# forbid: raise if the destination exists.
# resume: leave an existing file in place.
# force: overwrite after a warning via the artifact hook.
OverwritePolicy = str
ArtifactHook = Callable[[Path, bool], None]

_output_root: ContextVar[Path | None] = ContextVar("woais_output_root", default=None)
_overwrite_policy: ContextVar[str] = ContextVar("woais_overwrite_policy", default="replace")
_artifact_hook: ContextVar[ArtifactHook | None] = ContextVar("woais_artifact_hook", default=None)
_run_meta: ContextVar[dict[str, Any] | None] = ContextVar("woais_run_meta", default=None)


def get_results_root() -> Path:
    """Active write root. Timestamped run dirs when the CLI runner is active."""
    return _output_root.get() or RESULTS


def get_overwrite_policy() -> str:
    return _overwrite_policy.get() or "replace"


def set_overwrite_policy(policy: str) -> Any:
    """Switch the active write policy; returns a ContextVar token."""
    return _overwrite_policy.set(policy)


def get_artifact_hook() -> ArtifactHook | None:
    return _artifact_hook.get()


def get_run_meta() -> dict[str, Any] | None:
    meta = _run_meta.get()
    return None if meta is None else dict(meta)


def set_run_context(
    *,
    output_root: Path | None,
    overwrite_policy: str = "replace",
    artifact_hook: ArtifactHook | None = None,
    run_meta: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Install a run directory for `write_result`. Returns tokens for reset."""
    t1 = _output_root.set(output_root)
    t2 = _overwrite_policy.set(overwrite_policy)
    t3 = _artifact_hook.set(artifact_hook)
    t4 = _run_meta.set(run_meta)
    return t1, t2, t3, t4


def reset_run_context(tokens: tuple[Any, Any, Any, Any]) -> None:
    _output_root.reset(tokens[0])
    _overwrite_policy.reset(tokens[1])
    _artifact_hook.reset(tokens[2])
    _run_meta.reset(tokens[3])

FROZEN_DIRS = (
    ROOT / "raw_results",
    ROOT / "router_v2",
    ROOT / "stage7_10",
)
FROZEN_ROOT_FILES = (
    ROOT / "quality_benchmark_results.json",
    ROOT / "quality_benchmark_report.md",
    ROOT / "results_report.md",
)

LEGACY_IMPORT_DIRS = (
    ROOT / "benchmark",
    ROOT / "router_v2",
    ROOT / "stage7_10",
    ROOT / "backend",
)


def ensure_legacy_imports() -> None:
    """Enable `from api import MODELS` style imports used by existing scripts."""
    for path in LEGACY_IMPORT_DIRS:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def assert_inside_results(path: Path) -> Path:
    resolved = path.resolve()
    results = RESULTS.resolve()
    try:
        resolved.relative_to(results)
    except ValueError as exc:
        raise RuntimeError(f"refusing to write outside woais_experiments/results: {path}") from exc
    return resolved


def is_frozen(path: Path) -> bool:
    resolved = path.resolve()
    for frozen in FROZEN_DIRS:
        try:
            resolved.relative_to(frozen.resolve())
            return True
        except ValueError:
            continue
    return resolved in {p.resolve() for p in FROZEN_ROOT_FILES if p.exists()}
