"""YAML runner config and a stable SHA-256 over the files a run actually uses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from woais_experiments.paths import CONFIGS

DEFAULT_RUN_YAML = CONFIGS / "run.yaml"

CONFIG_FILES = (
    "run.yaml",
    "experiment.json",
    "models.json",
    "serverless.json",
    "workload_simulator.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping")
    return raw


def resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_RUN_YAML
    p = Path(path)
    if not p.is_absolute():
        if (CONFIGS / p).exists():
            p = CONFIGS / p
        else:
            p = Path.cwd() / p
    return p.resolve()


def load_run_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_config_path(path)
    cfg = load_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    exp_name = cfg.get("paths", {}).get("experiment_json", "experiment.json")
    exp_path = CONFIGS / exp_name
    if exp_path.exists():
        cfg["experiment"] = json.loads(exp_path.read_text())
    return cfg


def config_files_used(cfg: dict[str, Any] | None = None) -> list[Path]:
    names = list(CONFIG_FILES)
    if cfg:
        paths = cfg.get("paths") or {}
        extra = [
            paths.get("experiment_json"),
            paths.get("models_json"),
            paths.get("serverless_json"),
            paths.get("workload_yaml"),
        ]
        for name in extra:
            if name and name not in names:
                names.append(str(name))
    out: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        p = CONFIGS / str(name)
        if p.exists() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    run_yaml = Path(cfg["_config_path"]) if cfg and cfg.get("_config_path") else DEFAULT_RUN_YAML
    if run_yaml.exists() and run_yaml.resolve() not in seen:
        out.insert(0, run_yaml)
    return out


def config_hash(cfg: dict[str, Any] | None = None) -> str:
    """SHA-256 of concatenated config files (name + bytes). Not a secret."""
    h = hashlib.sha256()
    for path in config_files_used(cfg):
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


def seeds_from_config(cfg: dict[str, Any]) -> dict[str, int]:
    block = dict(cfg.get("seeds") or {})
    exp = cfg.get("experiment") or {}
    return {
        "random_policy": int(block.get("random_policy", exp.get("random_policy_seed", 20260905))),
        "monte_carlo": int(block.get("monte_carlo", exp.get("monte_carlo_seed", 20260909))),
        "arrival": int(block.get("arrival", exp.get("arrival_seed", 20260909))),
    }


def experiment_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shape expected by `run_offline` stage functions."""
    exp = dict(cfg.get("experiment") or {})
    seeds = seeds_from_config(cfg)
    boot = dict(cfg.get("bootstrap") or {})
    exp["random_policy_seed"] = seeds["random_policy"]
    exp["monte_carlo_seed"] = seeds["monte_carlo"]
    exp["arrival_seed"] = seeds["arrival"]
    exp["n_random_sims"] = int(boot.get("n_random_sims", exp.get("n_random_sims", 2000)))
    exp["api_calls"] = bool(cfg.get("api_calls", False))
    return exp
