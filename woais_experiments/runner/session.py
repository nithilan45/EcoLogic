"""Timestamped run directories, overwrite policy, and latest symlink."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from woais_experiments.frozen import sha256_file
from woais_experiments.paths import (
    CONFIGS,
    HASH_MANIFEST,
    RESULTS,
    ROOT,
    RUNS,
    public_relpath,
    set_run_context,
)
from woais_experiments.runner.config import config_hash, experiment_payload, seeds_from_config
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.runner.log import EventLogger, setup_logging

PAID_ENV = ("OPENAI_API_KEY", "TOGETHER_API_KEY", "ANTHROPIC_API_KEY")
STAGE_ORDER = (
    "audit",
    "accounting",
    "static",
    "oracle",
    "latency",
    "workload",
    "external",
    "robustness",
)
COMMANDS = STAGE_ORDER + ("all", "validate-artifact")


class PrerequisiteError(RuntimeError):
    """Environment or inputs are not ready."""


def utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(os.urandom(8)).hexdigest()[:6]
    return f"{stamp}_{suffix}"


def latest_link(runs_dir: Path = RUNS) -> Path:
    return runs_dir / "latest"


def resolve_resume_dir(value: str | None) -> Path | None:
    if value is None:
        return None
    if value in {"LATEST", "latest"}:
        link = latest_link()
        if link.is_symlink() or link.exists():
            return link.resolve()
        raise PrerequisiteError("no latest run directory to resume")
    p = Path(value)
    if p.exists():
        return p.resolve()
    if (RUNS / p).exists():
        return (RUNS / p).resolve()
    cand = Path.cwd() / p
    if cand.exists():
        return cand.resolve()
    return p.resolve()


def try_latest_symlink(run_dir: Path) -> bool:
    link = latest_link(run_dir.parent)
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
        return True
    except OSError:
        return False


def apply_seeds(seeds: dict[str, int]) -> None:
    random.seed(int(seeds["random_policy"]))
    np.random.seed(int(seeds["monte_carlo"]))


@dataclass
class RunSession:
    command: str
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    config_hash: str
    git: dict[str, Any]
    allow_api: bool
    overwrite: str
    seeds: dict[str, int]
    log: EventLogger
    verbose: bool = False
    symlink_ok: bool | None = None
    reset_tokens: tuple[Any, Any, Any, Any] | None = None
    matrix: Any = None
    routing: Any = None
    policies: Any = None
    exp: dict[str, Any] = field(default_factory=dict)
    srv: dict[str, Any] = field(default_factory=dict)
    summaries: dict[str, Any] = field(default_factory=dict)

    @property
    def run_meta(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git.get("commit"),
            "config_hash": self.config_hash,
        }

    def complete_path(self, stage: str) -> Path:
        return self.run_dir / "stages" / f"{stage}.complete.json"

    def is_complete(self, stage: str) -> bool:
        path = self.complete_path(stage)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return payload.get("status") == "ok"

    def mark_complete(self, stage: str, summary: Any, elapsed_s: float, status: str = "ok") -> None:
        dest = self.complete_path(stage)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "status": status,
            "elapsed_s": elapsed_s,
            "git_commit": self.git.get("commit"),
            "config_hash": self.config_hash,
            "summary": summary,
        }
        dest.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        self._record_file(dest, skipped=False)

    def _record_file(self, path: Path, skipped: bool) -> None:
        try:
            rel = path.resolve().relative_to(self.run_dir.resolve())
        except ValueError:
            rel = path
        rec = {
            "path": str(rel).replace("\\", "/"),
            "sha256": sha256_file(path) if path.exists() else None,
            "config_hash": self.config_hash,
            "git_commit": self.git.get("commit"),
            "skipped": skipped,
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        with (self.run_dir / "artifacts.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def index_tree(self) -> None:
        """Hash run-dir files not already listed in artifacts.jsonl."""
        listed: set[str] = set()
        art = self.run_dir / "artifacts.jsonl"
        if art.exists():
            for line in art.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    listed.add(json.loads(line)["path"])
                except (json.JSONDecodeError, KeyError):
                    continue
        skip_names = {"artifacts.jsonl", "run.log.jsonl"}
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file() or path.name in skip_names:
                continue
            try:
                rel = str(path.resolve().relative_to(self.run_dir.resolve())).replace("\\", "/")
            except ValueError:
                continue
            if rel in listed:
                continue
            self._record_file(path, skipped=False)

    def close(self) -> None:
        logger = logging.getLogger("woais")
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)
        if self.reset_tokens is not None:
            from woais_experiments.paths import reset_run_context
            reset_run_context(self.reset_tokens)
            self.reset_tokens = None


def _artifact_hook(session: RunSession) -> Callable[[Path, bool], None]:
    def hook(path: Path, skipped: bool) -> None:
        session._record_file(path, skipped)
    return hook


def open_session(
    *,
    command: str,
    config: dict[str, Any],
    allow_api: bool,
    resume_dir: Path | None,
    force: bool,
    run_id: str | None,
    no_symlink: bool,
    verbose: bool,
) -> RunSession:
    digest = config_hash(config)
    git = git_snapshot()
    seeds = seeds_from_config(config)

    if resume_dir is not None:
        run_dir = resume_dir
        if not run_dir.exists():
            raise PrerequisiteError(f"resume directory does not exist: {run_dir}")
        overwrite = "force" if force else "resume"
        rid = run_dir.name
    else:
        rid = run_id or utc_run_id()
        run_dir = RUNS / rid
        if run_dir.exists():
            raise PrerequisiteError(
                f"run directory exists: {run_dir}. Pass --resume or --force, "
                "or choose a new --run-id."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        overwrite = "force" if force else "forbid"

    log = setup_logging(run_dir, verbose=verbose)
    session = RunSession(
        command=command,
        run_id=rid,
        run_dir=run_dir,
        config=config,
        config_hash=digest,
        git=git,
        allow_api=allow_api,
        overwrite=overwrite,
        seeds=seeds,
        log=log,
        verbose=verbose,
        exp=experiment_payload(config),
    )
    session.reset_tokens = set_run_context(
        output_root=run_dir,
        overwrite_policy=overwrite,
        artifact_hook=_artifact_hook(session),
        run_meta=session.run_meta,
    )
    apply_seeds(seeds)
    if not no_symlink:
        session.symlink_ok = try_latest_symlink(run_dir)

    run_json = {
        "run_id": rid,
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "config_hash": digest,
        "config_path": (
            public_relpath(config["_config_path"]) if config.get("_config_path") else None
        ),
        "seeds": seeds,
        "allow_api": allow_api,
        "overwrite": overwrite,
        "python": sys.version,
        "results_canonical": public_relpath(RESULTS),
        "run_dir": public_relpath(run_dir),
    }
    run_json_path = run_dir / "run.json"
    cfg_path = run_dir / "config.yaml"
    if not run_json_path.exists() or overwrite == "force":
        run_json_path.write_text(json.dumps(run_json, indent=2) + "\n")
        session._record_file(run_json_path, skipped=False)
    if not cfg_path.exists() or overwrite == "force":
        cfg_src = Path(config["_config_path"]) if config.get("_config_path") else None
        cfg_path.write_text(
            cfg_src.read_text() if cfg_src and cfg_src.exists() else json.dumps(
                {k: v for k, v in config.items() if k != "experiment"}
            )
        )
        session._record_file(cfg_path, skipped=False)
    log.event(
        "run.start",
        run_id=rid,
        git_commit=git.get("commit"),
        config_hash=digest,
        allow_api=allow_api,
        overwrite=overwrite,
    )
    return session


def validate_prerequisites(session: RunSession, stage: str) -> list[str]:
    """Return error strings; empty means OK to run."""
    errors: list[str] = []
    py = session.config.get("python") or {}
    min_major = int(py.get("min_major", 3))
    min_minor = int(py.get("min_minor", 10))
    if sys.version_info < (min_major, min_minor):
        errors.append(
            f"Python {min_major}.{min_minor}+ required, running {sys.version.split()[0]}"
        )
    if session.config.get("write_outside_woais_experiments"):
        errors.append("write_outside_woais_experiments is not allowed")

    api_cfg = bool(session.config.get("api_calls"))
    if api_cfg and not session.allow_api:
        errors.append("config api_calls=true requires --allow-api")
    if session.allow_api and not api_cfg:
        session.log.event(
            "api.flag_without_config",
            level=logging.WARNING,
            note="--allow-api passed but YAML api_calls is false; paid calls stay disabled",
        )
    if not session.allow_api:
        present = [k for k in PAID_ENV if os.environ.get(k)]
        if present:
            session.log.event(
                "api.keys_present_unused",
                keys=present,
                note="paid keys are in the environment but will not be used",
            )

    required_files = [
        HASH_MANIFEST,
        ROOT / "raw_results" / "graded.jsonl",
        ROOT / "raw_results" / "routing.json",
        CONFIGS / "models.json",
        CONFIGS / "experiment.json",
        CONFIGS / "serverless.json",
        CONFIGS / "workload_simulator.yaml",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required file: {path}")

    if stage in {"external", "all"}:
        s9 = ROOT / "stage7_10" / "s9_static_baselines.json"
        if not s9.exists():
            errors.append(f"missing RouteLLM committed table: {s9}")

    want_hashes = True
    stage_cfg = (session.config.get("stages") or {}).get("audit") or {}
    if stage == "audit":
        want_hashes = bool(stage_cfg.get("require_frozen_hashes", True))
    if want_hashes and HASH_MANIFEST.exists():
        from woais_experiments.frozen import verify_frozen_hashes
        check = verify_frozen_hashes()
        if not check["ok"]:
            errors.append(
                f"frozen hash check failed: {check['n_mismatches']} mismatches, "
                f"{check['n_missing']} missing"
            )
    return errors
