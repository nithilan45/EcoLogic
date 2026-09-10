"""Publication-grade reproducibility and anonymization checks."""

from __future__ import annotations

__all__ = ["run_validate_artifact"]


def run_validate_artifact(**kwargs):
    from woais_experiments.reproducibility.validate import run_validate_artifact as _run

    return _run(**kwargs)
