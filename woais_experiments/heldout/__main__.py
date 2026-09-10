"""Run the held-out protocol: freeze split → train/val select → one-shot test."""

from __future__ import annotations

import argparse

from woais_experiments.heldout.build_split import build_manifest
from woais_experiments.heldout.common import LockedExperimentError, N_BOOT, N_PERM
from woais_experiments.heldout.evaluate_router import evaluate
from woais_experiments.heldout.train_router import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Held-out EcoLogic routing protocol.")
    p.add_argument("--force-split", action="store_true", help="Rebuild split_manifest.json")
    p.add_argument("--new-experiment", action="store_true")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_manifest(force=bool(args.force_split))
    try:
        train(new_experiment=bool(args.new_experiment))
        payload = evaluate(
            new_experiment=bool(args.new_experiment),
            n_boot=int(args.n_boot),
            n_perm=int(args.n_perm),
        )
    except LockedExperimentError as exc:
        print(exc)
        return 1
    from woais_experiments.heldout.analyze_heldout import print_summary

    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
