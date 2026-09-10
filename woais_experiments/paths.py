"""Repo paths and a one-way door: frozen trees are read-only from this package."""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
RESULTS = PACKAGE / "results"
RUNS = RESULTS / "runs"
CONFIGS = PACKAGE / "configs"
HASH_MANIFEST = PACKAGE / "EXISTING_RESULTS_SHA256.txt"

REDACTED_ABSOLUTE = "<redacted-absolute>"


def _is_windows_path_string(text: str) -> bool:
    """True for drive-letter or UNC path syntax. Not a host-prefix allowlist."""
    s = str(text)
    if len(s) >= 3 and s[0].isalpha() and s[1] == ":" and s[2] in "/\\":
        return True
    return s.startswith("\\\\")


def is_absolute_path_string(text: str) -> bool:
    """True for POSIX or Windows absolute filesystem path *syntax*."""
    s = str(text).strip()
    if not s or "\n" in s:
        return False
    if s.startswith("/") and not s.startswith("//"):
        return True
    if s.startswith("\\\\"):
        return True
    return _is_windows_path_string(s)


def _try_resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return path


def relative_to_root(path: Path | str, root: Path | str) -> str | None:
    """POSIX-relative path if ``path`` is inside ``root``, else None.

    Uses pathlib relationship checks only. Does not consult a host-prefix list.
    """
    path_s = str(path)
    root_s = str(root)
    if _is_windows_path_string(path_s) or _is_windows_path_string(root_s):
        try:
            rel = PureWindowsPath(path_s).relative_to(PureWindowsPath(root_s))
        except ValueError:
            return None
        posix = rel.as_posix()
        if posix == ".":
            return "."
        if posix.startswith("../") or posix == "..":
            return None
        return posix

    root_p = root if isinstance(root, Path) else Path(root_s)
    path_p = path if isinstance(path, Path) else Path(path_s)
    if not path_p.is_absolute():
        path_p = root_p / path_p

    pairs: list[tuple[PurePath, PurePath]] = []
    resolved_root = _try_resolve(root_p)
    resolved_path = _try_resolve(path_p)
    pairs.append((resolved_path, resolved_root))
    pairs.append((PurePosixPath(path_p.as_posix()), PurePosixPath(root_p.as_posix())))
    pairs.append((
        PurePosixPath(resolved_path.as_posix()),
        PurePosixPath(resolved_root.as_posix()),
    ))

    seen: set[str] = set()
    for p, r in pairs:
        key = f"{p.as_posix()}\0{r.as_posix()}"
        if key in seen:
            continue
        seen.add(key)
        try:
            rel = p.relative_to(r)
        except ValueError:
            continue
        posix = rel.as_posix()
        if posix.startswith("../") or posix == "..":
            continue
        return "." if posix == "." else posix
    return None


def repo_rel(path: Path | str, *, root: Path | str | None = None) -> str:
    """Path relative to the repo root when ``path`` is inside that root."""
    base = ROOT if root is None else root
    rel = relative_to_root(path, base)
    if rel is not None:
        return rel
    if isinstance(path, Path):
        return path.as_posix()
    return str(path).replace("\\", "/")


def looks_like_home_absolute(text: str, *, root: Path | str | None = None) -> bool:
    """True for absolute paths that are *not* inside the repository.

    Kept for callers that redact author-machine locations. Detection is
    pathlib containment, not a list of ``/Users`` / ``/home`` prefixes.
    """
    if not is_absolute_path_string(text):
        return False
    return relative_to_root(text, ROOT if root is None else root) is None


def public_relpath(path: Path | str, *, root: Path | str | None = None) -> str:
    """Serialize a filesystem path for artifacts.

    In-repo paths become repository-relative regardless of the absolute parent
    (``/Users``, ``/home``, ``/mnt/data``, ``/tmp``, Windows drive letters).
    Paths outside the repository are anonymized.
    """
    base = ROOT if root is None else root
    rel = relative_to_root(path, base)
    if rel is not None:
        return rel
    return REDACTED_ABSOLUTE

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
