"""Inventory, hashes, and package-fitness checks for a release tree."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from woais_experiments.frozen import parse_manifest, sha256_file
from woais_experiments.paths import ROOT

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".cursor",
    "hf_cache",
    "runs",
    "_stress_unittest",
}

LARGE_BYTES = 5 * 1024 * 1024
REQUIRED_LOCK = "woais_experiments/requirements-lock.txt"


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def tracked_or_walk(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else ROOT
    frozen = {r.relpath: {"sha256": r.sha256, "size": r.size} for r in parse_manifest()}
    rows = []
    for path in sorted(tracked_or_walk(base)):
        rel = relative(path, base)
        try:
            size = path.stat().st_size
            digest = file_sha256(path) if size <= 80 * 1024 * 1024 else None
        except OSError:
            continue
        rows.append({
            "path": rel,
            "bytes": size,
            "sha256": digest,
            "frozen": rel in frozen,
        })
    return {
        "n_files": len(rows),
        "files": rows,
        "n_frozen_manifest": len(frozen),
    }


def package_check(root: Path | None = None) -> dict[str, Any]:
    """Flag files that usually should not ship in an anonymous reproduction artifact."""
    base = Path(root) if root is not None else ROOT
    frozen = {r.relpath for r in parse_manifest()}
    warnings: list[dict[str, Any]] = []
    critical: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    notebooks = []
    checkpoints = []
    large = []
    for path in tracked_or_walk(base):
        rel = relative(path, base)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        low = rel.lower()
        if path.name == "secret_scan.py":
            continue
        if path.name == ".env.example" or path.name.endswith(".env.example"):
            warnings.append({"path": rel, "reason": "env_template"})
            continue
        if path.name in {".env", ".env.local", ".env.rc"} or path.name.endswith("credentials.json"):
            critical.append({"path": rel, "reason": "private_or_credential_filename"})
        if path.name in {"id_rsa", "id_ed25519", "id_ecdsa"} or path.suffix.lower() in {".pem", ".p12", ".key"}:
            critical.append({"path": rel, "reason": "private_or_credential_filename"})
        if path.suffix == ".ipynb":
            notebooks.append(rel)
            warnings.append({"path": rel, "reason": "notebook"})
        if any(tok in low for tok in ("checkpoint", ".ckpt", ".pt", "cache/http", "api_cache")):
            if rel not in frozen:
                checkpoints.append(rel)
                warnings.append({"path": rel, "reason": "checkpoint_or_cache"})
        if size >= LARGE_BYTES and rel not in frozen:
            large.append({"path": rel, "bytes": size})
            warnings.append({"path": rel, "reason": "large_unfrozen_file", "bytes": size})
        if "hf_cache" in rel.replace("\\", "/"):
            warnings.append({"path": rel, "reason": "huggingface_cache"})
        if size <= 80 * 1024 * 1024:
            try:
                hashes[file_sha256(path)].append(rel)
            except OSError:
                pass
    duplicates = []
    for digest, paths in hashes.items():
        if len(paths) < 2:
            continue
        # Ignore identical tiny files; flag duplicate bulky outputs.
        if all(p.endswith((".py", ".md", ".txt", ".yaml")) for p in paths):
            continue
        duplicates.append({"sha256": digest, "paths": paths})
        warnings.append({"path": paths[0], "reason": "duplicate_output", "paths": paths})

    lock = base / REQUIRED_LOCK
    if not lock.exists():
        critical.append({"path": REQUIRED_LOCK, "reason": "missing_lockfile"})

    return {
        "ok": len(critical) == 0,
        "n_warnings": len(warnings),
        "n_critical": len(critical),
        "critical": critical,
        "warnings": warnings,
        "notebooks": notebooks,
        "checkpoints": checkpoints,
        "large_unfrozen": large,
        "duplicates": duplicates,
        "note": (
            "Frozen hashed files are treated as required. Warnings are not "
            "automatic deletions."
        ),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
