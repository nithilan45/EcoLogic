"""Score the frozen held-out configuration on TEST exactly once.

Train is used only to refit logistic/tree at frozen hyperparameters.
Validation is not consulted. Test metrics are not used to select anything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.costs import TIERS, cost_fn_usd, evaluate_assignment
from woais_experiments.frozen import write_result
from woais_experiments.heldout.build_split import (
    load_manifest,
    load_panel,
    split_ids,
    subset,
    verify_manifest_hash,
)
from woais_experiments.heldout.cluster_stats import cluster_mean_ci, cluster_paired_mean
from woais_experiments.heldout.common import (
    FINAL_CSV_REL,
    FROZEN_CONFIG_REL,
    HeldoutLeakageError,
    LOCK_REL,
    LockedExperimentError,
    N_BOOT,
    N_PERM,
    QUERY_CSV_REL,
    assert_same_test_population,
    hash_config,
    mix_int,
    quota_assignment_sorted,
    refuse_if_locked,
    rows_to_csv,
    sanitize,
    protocol_source_sha256,
)
from woais_experiments.heldout.train_router import (
    dense_matrix,
    fit_train_preprocessor,
    heuristic_assign,
    length_assign,
    oracle_labels,
    predict_int,
    refit_logistic,
    refit_tree,
    texts_for,
)
from woais_experiments.paths import get_results_root
from woais_experiments.routing.ablations import _lexicons, random_matched_assignment
from woais_experiments.routing.cost_matching import compare_router_to_static
from woais_experiments.routing.policies import oracle as oracle_assignment
from woais_experiments.routing.static_baselines import market_from_panel

METHOD_ORDER = (
    "always_cheap",
    "always_strong",
    "random_matched",
    "cost_matched_static",
    "logistic",
    "tree",
    "threshold",
    "ecologic_heuristic",
    "oracle",
)

ORACLE_METHODS = {"oracle"}
BASELINE_METHODS = {"always_cheap", "always_strong", "random_matched", "cost_matched_static"}

FINAL_FIELDS = (
    "method",
    "role",
    "selected",
    "n",
    "quality",
    "realized_cost",
    "quality_advantage_vs_cost_matched_static",
    "oracle_gap",
    "strong_model_fraction",
    "quality_at_same_realized_cost",
    "cost_at_same_quality",
    "in_sample_hull_quality_advantage",
    "quality_ci_lo",
    "quality_ci_hi",
    "cost_ci_lo",
    "cost_ci_hi",
    "advantage_ci_lo",
    "advantage_ci_hi",
    "advantage_p_raw",
    "cohens_dz_vs_cost_matched_static",
    "cliffs_delta_vs_cost_matched_static",
    "bootstrap_unit",
    "n_groups",
    "n_boot",
    "n_perm",
)

QUERY_FIELDS = (
    "item_id",
    "group_id",
    "benchmark",
    "method",
    "tier",
    "correct",
    "usd",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "selected_method",
)


def load_frozen_config() -> dict[str, Any]:
    path = get_results_root() / FROZEN_CONFIG_REL
    if not path.exists():
        raise FileNotFoundError(f"frozen config missing: {path}. Run train_router.py first.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    stored = cfg.get("config_sha256")
    recomputed = hash_config(cfg)
    if stored != recomputed:
        raise HeldoutLeakageError("frozen_config.json hash mismatch — file was edited after freeze")
    if cfg.get("test_metrics_inspected"):
        raise HeldoutLeakageError("frozen_config claims test metrics were inspected during training")
    return cfg


def group_of_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {str(q["item_id"]): str(q["group_id"]) for q in manifest["queries"]}


def build_test_assignments(
    config: Mapping[str, Any],
    matrix,
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    cost_of,
    *,
    seed: int,
) -> dict[str, dict[str, int]]:
    lex = _lexicons()
    train_text = texts_for(records, train_ids)
    test_text = texts_for(records, test_ids)
    x_tr_dense = dense_matrix(train_text, lex=lex)
    x_te_dense = dense_matrix(test_text, lex=lex)
    feats = fit_train_preprocessor(train_text, x_tr_dense, seed=seed)
    x_tr, _, _ = feats.transform(train_text, x_tr_dense)
    x_te, _, _ = feats.transform(test_text, x_te_dense)
    y_tr = oracle_labels(matrix, train_ids, cost_of)

    log_clf = refit_logistic(x_tr, y_tr, C=config["logistic"].get("C"), seed=seed)
    tree_clf = refit_tree(
        x_tr_dense,
        y_tr,
        max_depth=config["tree"].get("max_depth"),
        learning_rate=config["tree"].get("learning_rate"),
        seed=seed,
    )
    cheap = int(config["cheap_tier"])
    strong = int(config["strong_tier"])
    assigns: dict[str, dict[str, int]] = {
        "logistic": predict_int(log_clf, x_te, test_ids),
        "tree": predict_int(tree_clf, x_te_dense, test_ids),
        "threshold": length_assign(
            test_ids,
            records,
            tau=float(config["threshold"]["tau"]),
            cheap=cheap,
            strong=strong,
            lex=lex,
        ),
        "ecologic_heuristic": heuristic_assign(test_ids, records, lex),
        "always_cheap": {str(i): cheap for i in test_ids},
        "always_strong": {str(i): strong for i in test_ids},
        "oracle": oracle_assignment(subset(matrix, test_ids), cost_of),
        "random_matched": random_matched_assignment(
            sorted(test_ids),
            mix_int(config["random_matched_mix"]["weights"]),
            seed=int(seed),
        ),
        "cost_matched_static": quota_assignment_sorted(
            test_ids, mix_int(config["cost_matched_static"]["weights"])
        ),
    }
    return assigns


def _role(name: str) -> str:
    if name in ORACLE_METHODS:
        return "hindsight_oracle"
    if name in BASELINE_METHODS:
        return "baseline"
    return "deployable"


def _series(
    matrix, ids: Sequence[str], assign: Mapping[str, int], cost_of
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([float(bool(matrix.correct[(int(assign[i]), i)])) for i in ids], dtype=float)
    c = np.asarray([float(cost_of(int(assign[i]), i)) for i in ids], dtype=float)
    return q, c


def evaluate(
    *,
    new_experiment: bool = False,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> dict[str, Any]:
    previous = refuse_if_locked(new_experiment=new_experiment, action="re-evaluate the held-out test set")
    manifest = load_manifest()
    verify_manifest_hash(manifest)
    config = load_frozen_config()
    if config.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
        raise HeldoutLeakageError("frozen_config split hash does not match split_manifest.json")

    splits = split_ids(manifest)
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise HeldoutLeakageError("test overlaps train/val")

    matrix, records = load_panel()
    cost_of = cost_fn_usd(matrix)
    group_of = group_of_from_manifest(manifest)
    seed = int(config["seed"])
    assigns = build_test_assignments(
        config, matrix, records, train_ids, test_ids, cost_of, seed=seed
    )
    assert_same_test_population(assigns, test_ids)

    test_m = subset(matrix, test_ids)
    cost_panel = np.asarray([[cost_of(t, i) for t in TIERS] for i in test_ids], dtype=float)
    qual_panel = np.asarray(
        [[float(bool(matrix.correct[(t, i)])) for t in TIERS] for i in test_ids], dtype=float
    )
    hull = market_from_panel([str(t) for t in TIERS], cost_panel, qual_panel)

    q_static, c_static = _series(matrix, test_ids, assigns["cost_matched_static"], cost_of)
    q_oracle, _ = _series(matrix, test_ids, assigns["oracle"], cost_of)
    strong = int(config["strong_tier"])

    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    for name in METHOD_ORDER:
        assign = assigns[name]
        q, c = _series(matrix, test_ids, assign, cost_of)
        scored = evaluate_assignment(test_m, assign, cost_of, name=name)
        vs_hull = compare_router_to_static(hull, float(q.mean()), float(c.mean()))
        adv = float(q.mean() - q_static.mean())
        oracle_gap = float(q_oracle.mean() - q.mean())
        q_ci = cluster_mean_ci(
            q, ids=test_ids, group_of=group_of, n_boot=n_boot, seed=seed, name=f"{name}_quality"
        )
        c_ci = cluster_mean_ci(
            c, ids=test_ids, group_of=group_of, n_boot=n_boot, seed=seed + 1, name=f"{name}_cost"
        )
        paired = cluster_paired_mean(
            q,
            q_static,
            ids=test_ids,
            group_of=group_of,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 2,
            name=f"{name}_minus_cost_matched_static",
        )
        dz = (paired.get("effect_size") or {}).get("cohens_dz") or {}
        cliff = (paired.get("effect_size") or {}).get("cliffs_delta") or {}
        row = {
            "method": name,
            "role": _role(name),
            "selected": int(name == config["selected_method"]),
            "n": len(test_ids),
            "quality": float(q.mean()),
            "realized_cost": float(c.mean()),
            "quality_advantage_vs_cost_matched_static": adv,
            "oracle_gap": oracle_gap,
            "strong_model_fraction": float(sum(1 for i in test_ids if int(assign[i]) == strong) / len(test_ids)),
            "quality_at_same_realized_cost": vs_hull.get("static_quality_at_same_cost"),
            "cost_at_same_quality": vs_hull.get("static_cost_at_same_quality"),
            "in_sample_hull_quality_advantage": vs_hull.get("quality_advantage"),
            "quality_ci_lo": q_ci.get("ci_lo"),
            "quality_ci_hi": q_ci.get("ci_hi"),
            "cost_ci_lo": c_ci.get("ci_lo"),
            "cost_ci_hi": c_ci.get("ci_hi"),
            "advantage_ci_lo": paired.get("ci_lo"),
            "advantage_ci_hi": paired.get("ci_hi"),
            "advantage_p_raw": paired.get("p_raw"),
            "cohens_dz_vs_cost_matched_static": dz.get("estimate"),
            "cliffs_delta_vs_cost_matched_static": cliff.get("estimate"),
            "bootstrap_unit": "group",
            "n_groups": paired.get("n_groups"),
            "n_boot": int(n_boot),
            "n_perm": int(n_perm),
            "in_sample_hull_note": (
                "quality_at_same_realized_cost and cost_at_same_quality interpolate the "
                "in-sample TEST static hull; they are descriptive, not a pre-registered policy. "
                "cost_matched_static is the VAL-frozen mix applied to TEST."
            ),
            "correct": scored["correct"],
            "tier_mix": scored["tier_mix"],
        }
        rows.append(row)
        stats[name] = {
            "quality": q_ci,
            "cost": c_ci,
            "vs_cost_matched_static": paired,
            "in_sample_hull": vs_hull,
        }
        for i in test_ids:
            t = int(assign[i])
            query_rows.append({
                "item_id": i,
                "group_id": group_of[i],
                "benchmark": records[i]["benchmark"],
                "method": name,
                "tier": t,
                "correct": int(bool(matrix.correct[(t, i)])),
                "usd": float(cost_of(t, i)),
                "prompt_tokens": int(matrix.prompt_tokens[(t, i)]),
                "completion_tokens": int(matrix.completion_tokens[(t, i)]),
                "total_tokens": int(matrix.tokens[(t, i)]),
                "selected_method": config["selected_method"],
            })

    write_result(FINAL_CSV_REL, rows_to_csv(rows, FINAL_FIELDS), clobber=True)
    write_result(QUERY_CSV_REL, rows_to_csv(query_rows, QUERY_FIELDS), clobber=True)

    lock = {
        "protocol": "heldout_v1",
        "lock": True,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_sha256": config["config_sha256"],
        "split_manifest_sha256": manifest["manifest_sha256"],
        "test_ids_hash": config["test_ids_hash"],
        "n_test": len(test_ids),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "selected_method": config["selected_method"],
        "bootstrap_unit": "group",
        "protocol_source_sha256": protocol_source_sha256(),
        "new_experiment": bool(new_experiment),
        "previous_lock_present": previous is not None,
        "note": (
            "Test was scored once after configuration freeze. Do not retune. "
            "Pass --new-experiment to start a different run; that does not un-see this evaluation."
        ),
        "historical_stage12_n364_not_used": True,
    }
    if previous is not None:
        lock["previous_lock_sha256"] = hash_config(previous)
    write_result(LOCK_REL, sanitize(lock), clobber=True)
    return {
        "n": dict(manifest["n"]),
        "selected_method": config["selected_method"],
        "rows": rows,
        "stats": stats,
        "lock": lock,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate frozen held-out routers on TEST once.")
    p.add_argument("--new-experiment", action="store_true")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
