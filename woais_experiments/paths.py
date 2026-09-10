"""Repo paths and a one-way door: frozen trees are read-only from this package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
RESULTS = PACKAGE / "results"
CONFIGS = PACKAGE / "configs"
HASH_MANIFEST = PACKAGE / "EXISTING_RESULTS_SHA256.txt"

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
