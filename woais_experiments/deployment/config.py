"""YAML loader for the deployment benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from woais_experiments.paths import PACKAGE

DEFAULT_CONFIG = PACKAGE / "deployment" / "deployment_config.yaml"


def load_deployment_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"deployment config {p} must be a mapping")
    raw["_config_path"] = str(p.resolve())
    return raw
