"""Run the router ablation study and write MEASURED-style tables.

Fit/tune never see the test split. Outputs:

- woais_experiments/results/ablations/summary.csv
- woais_experiments/results/ablations/query_level.csv
- woais_experiments/results/ablations/configs.json
"""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from typing import Any, Mapping

from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath
from woais_experiments.routing.ablations import (
    DEFAULT_N_BOOT,
    DEFAULT_N_PERM,
    MODEL_SEED,
    SPLIT_SEED,
    TestQueryMismatchError,
    assert_same_test_queries,
    query_level_rows,
    run_ablation_study,
    summary_rows,
)

PREFIX = "ablations"


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def write_ablation_outputs(study: Mapping[str, Any], *, prefix: str = PREFIX) -> dict[str, str]:
    assert_same_test_queries(study["variants"], study["test_ids"])
    configs = dict(study["configs"])
    configs["test_queries_identical"] = True
    configs["select_ablation_on_test"] = False
    summary = summary_rows(study)
    queries = query_level_rows(study)
    # Every variant must appear, including those that lose to the full router.
    paths = {
        "summary_csv": public_relpath(write_result(f"{prefix}/summary.csv", _csv(summary))),
        "query_level_csv": public_relpath(write_result(f"{prefix}/query_level.csv", _csv(queries))),
        "configs_json": public_relpath(write_result(f"{prefix}/configs.json", configs)),
    }
    index = {
        "n_variants": len(study["variants"]),
        "n_test": len(study["test_ids"]),
        "variants": sorted(study["variants"]),
        "paths": paths,
        "select_ablation_on_test": False,
        "test_queries_identical": True,
        "negative_results_preserved": True,
    }
    paths["index_json"] = public_relpath(write_result(f"{prefix}/index.json", index))
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EcoLogic router ablation study (train/val/test).")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--model-seed", type=int, default=MODEL_SEED)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--dry-run", action="store_true", help="Run the study but do not write results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study = run_ablation_study(
        n_boot=int(args.n_boot),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        model_seed=int(args.model_seed),
    )
    try:
        assert_same_test_queries(study["variants"], study["test_ids"])
    except TestQueryMismatchError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "n_variants": len(study["variants"]),
            "n_test": len(study["test_ids"]),
            "strongest_family": study["configs"]["strongest_family"],
            "wrote": False,
        }, indent=2))
        return 0
    paths = write_ablation_outputs(study, prefix=str(args.prefix))
    print(json.dumps({"ok": True, "paths": paths, "n_test": len(study["test_ids"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
