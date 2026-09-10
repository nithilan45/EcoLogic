"""Capture the runtime used to reproduce WOAIS results. No secrets."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from woais_experiments.frozen import sha256_file
from woais_experiments.paths import (
    HASH_MANIFEST,
    PACKAGE,
    RESULTS,
    ROOT,
    looks_like_home_absolute,
    public_relpath,
    repo_rel,
)
from woais_experiments.runner.config import config_hash, load_run_config, seeds_from_config
from woais_experiments.runner.gitinfo import git_snapshot

PAID_ENV = ("OPENAI_API_KEY", "TOGETHER_API_KEY", "ANTHROPIC_API_KEY")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def python_info() -> dict[str, Any]:
    return {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "executable": (
            public_relpath(sys.executable)
            if looks_like_home_absolute(sys.executable)
            else sys.executable
        ),
        "version_full": sys.version,
    }


def platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def dependency_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn",
        "yaml",
        "httpx",
        "datasets",
        "pandas",
    )
    out: dict[str, str | None] = {}
    for name in names:
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", None)
        except Exception:
            out[name] = None
    return out


def paid_keys_present() -> list[str]:
    """Names only. Never values."""
    return [k for k in PAID_ENV if os.environ.get(k)]


def _hash_if_exists(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def artifact_hashes() -> dict[str, str | None]:
    files = (
        HASH_MANIFEST,
        PACKAGE / "requirements-lock.txt",
        RESULTS / "accounting" / "stage12.json",
        RESULTS / "routing" / "stage12_policies.json",
        RESULTS / "external" / "routellm_tables.json",
    )
    return {repo_rel(path): _hash_if_exists(path) for path in files}


def capture_environment(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_run_config()
    seeds = seeds_from_config(cfg)
    git = git_snapshot()
    manifest_hash = _hash_if_exists(HASH_MANIFEST)
    payload = {
        "captured_utc": utc_now(),
        "python": python_info(),
        "platform": platform_info(),
        "dependencies": dependency_versions(),
        "git": git,
        "random_seeds": seeds,
        "config_hash": config_hash(cfg),
        "frozen_manifest_sha256": manifest_hash,
        "artifact_hashes": artifact_hashes(),
        "paid_api_key_names_present": paid_keys_present(),
        "cwd_is_repo": Path.cwd().resolve() == ROOT.resolve(),
    }
    if extra:
        payload.update(extra)
    return payload
