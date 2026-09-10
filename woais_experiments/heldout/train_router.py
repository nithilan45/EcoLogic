"""Fit candidate routers on TRAIN; select hyperparameters on VALIDATION only.

Never reads test labels, never fits on test text, never inspects test metrics.
Writes ``woais_experiments/results/heldout/frozen_config.json``.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from woais_experiments.accounting.costs import TIERS, cost_fn_usd
from woais_experiments.frozen import write_result
from woais_experiments.heldout.build_split import load_manifest, load_panel, split_ids, subset, verify_manifest_hash
from woais_experiments.heldout.common import (
    ConstantClassifier,
    FROZEN_CONFIG_REL,
    HeldoutLeakageError,
    LockedExperimentError,
    MODEL_SEED,
    as_dense,
    hash_config,
    refuse_if_locked,
    sanitize,
)
from woais_experiments.routing.ablations import (
    DENSE_NAMES,
    TrainOnlyFeatures,
    _lexicons,
    dense_features,
    gated_ecologic_tier,
)
from woais_experiments.routing.cost_matching import compare_router_to_static
from woais_experiments.routing.frontier import interpolate_at_cost
from woais_experiments.routing.policies import oracle as oracle_assignment
from woais_experiments.routing.static_baselines import market_from_panel

LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
TREE_DEPTH = (2, 3, 4)
TREE_LR = (0.05, 0.1)
TFIDF_MAX_FEATURES = 512

SELECTION_CRITERION = (
    "On validation only: maximize quality_advantage versus the in-sample "
    "cost-matched static hull. If that advantage is undefined, maximize "
    "validation quality and break ties by lower realized USD. "
    "Candidates are logistic, boosted tree, length threshold, and the frozen "
    "production EcoLogic heuristic. Baselines are not selected. "
    "The test split is not used for fitting, thresholding, or selection."
)


def texts_for(records: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> list[str]:
    return [str(records[i]["raw_query"]) for i in ids]


def dense_matrix(texts: Sequence[str], *, lex: dict[str, Any] | None = None) -> np.ndarray:
    lex = lex or _lexicons()
    rows = [dense_features(t, lex=lex) for t in texts]
    return np.asarray([[float(r[n]) for n in DENSE_NAMES] for r in rows], dtype=float)


def oracle_labels(matrix, ids: Sequence[str], cost_of) -> np.ndarray:
    assign = oracle_assignment(subset(matrix, ids), cost_of)
    return np.asarray([int(assign[i]) for i in ids], dtype=int)


def mean_cost_quality(matrix, ids: Sequence[str], cost_of) -> tuple[dict[int, float], dict[int, float]]:
    n = len(ids)
    cost = {t: sum(cost_of(t, i) for i in ids) / n for t in TIERS}
    qual = {t: sum(bool(matrix.correct[(t, i)]) for i in ids) / n for t in TIERS}
    return cost, qual


def cheap_strong_from_train(mean_cost: Mapping[int, float]) -> tuple[int, int]:
    cheap = min(TIERS, key=lambda t: (mean_cost[t], t))
    strong = max(TIERS, key=lambda t: (mean_cost[t], -t))
    return int(cheap), int(strong)


def assignment_quality_cost(
    matrix, ids: Sequence[str], assign: Mapping[str, int], cost_of
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([float(bool(matrix.correct[(int(assign[i]), i)])) for i in ids], dtype=float)
    c = np.asarray([float(cost_of(int(assign[i]), i)) for i in ids], dtype=float)
    return q, c


def val_quality_advantage(matrix, ids: Sequence[str], assign: Mapping[str, int], cost_of) -> float | None:
    cost = np.asarray([[cost_of(t, i) for t in TIERS] for i in ids], dtype=float)
    quality = np.asarray([[float(bool(matrix.correct[(t, i)])) for t in TIERS] for i in ids], dtype=float)
    q, c = assignment_quality_cost(matrix, ids, assign, cost_of)
    market = market_from_panel([str(t) for t in TIERS], cost, quality)
    vs = compare_router_to_static(market, float(q.mean()), float(c.mean()))
    adv = vs.get("quality_advantage")
    return None if adv is None else float(adv)


def score_tuple(adv: float | None, quality: float, cost: float) -> tuple[int, float, float, float]:
    """Larger is better. Undefined advantage ranks below a defined one."""
    has = 1 if adv is not None else 0
    return (has, float(adv if adv is not None else 0.0), float(quality), -float(cost))


def heuristic_assign(ids: Sequence[str], records: Mapping[str, Mapping[str, Any]], lex: dict[str, Any]) -> dict[str, int]:
    return {i: int(gated_ecologic_tier(records[i]["raw_query"], lex=lex)) for i in ids}


def length_assign(
    ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    *,
    tau: float,
    cheap: int,
    strong: int,
    lex: dict[str, Any],
) -> dict[str, int]:
    out = {}
    for i in ids:
        f = dense_features(records[i]["raw_query"], lex=lex)
        out[i] = int(strong) if float(f["log_n_words"]) >= float(tau) else int(cheap)
    return out


def select_length_tau(
    val_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    matrix,
    cost_of,
    *,
    cheap: int,
    strong: int,
    lex: dict[str, Any],
) -> dict[str, Any]:
    """Threshold chosen on validation only. ``val_ids`` must not include test."""
    lengths = np.asarray(
        [dense_features(records[i]["raw_query"], lex=lex)["log_n_words"] for i in val_ids],
        dtype=float,
    )
    candidates = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, 21)))
    best: dict[str, Any] | None = None
    best_key = None
    for tau in candidates:
        assign = length_assign(val_ids, records, tau=float(tau), cheap=cheap, strong=strong, lex=lex)
        q, c = assignment_quality_cost(matrix, val_ids, assign, cost_of)
        adv = val_quality_advantage(matrix, val_ids, assign, cost_of)
        key = score_tuple(adv, float(q.mean()), float(c.mean()))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "tau": float(tau),
                "cheap": int(cheap),
                "strong": int(strong),
                "feature": "log_n_words",
                "val_quality": float(q.mean()),
                "val_cost": float(c.mean()),
                "val_quality_advantage": adv,
                "selected_on": "validation",
                "test_used": False,
            }
    assert best is not None
    return best


def predict_int(clf, x, ids: Sequence[str]) -> dict[str, int]:
    try:
        pred = clf.predict(x)
    except (TypeError, ValueError):
        pred = clf.predict(as_dense(x))
    classes = {int(c) for c in np.asarray(clf.classes_).tolist()}
    fallback = int(next(iter(classes)))
    out = {}
    for i, y in zip(ids, pred):
        y = int(y)
        out[str(i)] = y if y in classes else fallback
    return out


def _maybe_constant(y: np.ndarray) -> ConstantClassifier | None:
    uniq = np.unique(y)
    if uniq.size < 2:
        return ConstantClassifier(int(uniq[0]))
    return None


def fit_logistic_grid(
    x_train,
    y_train: np.ndarray,
    x_val,
    val_ids: Sequence[str],
    matrix,
    cost_of,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    const = _maybe_constant(y_train)
    x_tr, x_va = as_dense(x_train), as_dense(x_val)
    if const is not None:
        const.fit(x_tr, y_train)
        assign = predict_int(const, x_va, val_ids)
        q, cost = assignment_quality_cost(matrix, val_ids, assign, cost_of)
        adv = val_quality_advantage(matrix, val_ids, assign, cost_of)
        return const, {
            "C": None,
            "degenerate": True,
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": [1, 2],
            "val_quality": float(q.mean()),
            "val_cost": float(cost.mean()),
            "val_quality_advantage": adv,
            "selected_on": "validation",
            "test_used": False,
        }
    best_clf = None
    best_meta: dict[str, Any] | None = None
    best_key = None
    for c in LOGISTIC_C:
        clf = LogisticRegression(
            C=float(c),
            max_iter=800,
            solver="lbfgs",
            random_state=int(seed),
        )
        clf.fit(x_tr, y_train)
        assign = predict_int(clf, x_va, val_ids)
        q, cost = assignment_quality_cost(matrix, val_ids, assign, cost_of)
        adv = val_quality_advantage(matrix, val_ids, assign, cost_of)
        key = score_tuple(adv, float(q.mean()), float(cost.mean()))
        if best_key is None or key > best_key:
            best_key = key
            best_clf = clf
            best_meta = {
                "C": float(c),
                "degenerate": False,
                "max_features": TFIDF_MAX_FEATURES,
                "ngram_range": [1, 2],
                "val_quality": float(q.mean()),
                "val_cost": float(cost.mean()),
                "val_quality_advantage": adv,
                "selected_on": "validation",
                "test_used": False,
            }
    assert best_clf is not None and best_meta is not None
    return best_clf, best_meta


def fit_tree_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val_ids: Sequence[str],
    matrix,
    cost_of,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    const = _maybe_constant(y_train)
    x_tr, x_va = as_dense(x_train), as_dense(x_val)
    if const is not None:
        const.fit(x_tr, y_train)
        assign = predict_int(const, x_va, val_ids)
        q, cost = assignment_quality_cost(matrix, val_ids, assign, cost_of)
        adv = val_quality_advantage(matrix, val_ids, assign, cost_of)
        return const, {
            "max_depth": None,
            "learning_rate": None,
            "max_iter": 120,
            "degenerate": True,
            "val_quality": float(q.mean()),
            "val_cost": float(cost.mean()),
            "val_quality_advantage": adv,
            "selected_on": "validation",
            "test_used": False,
        }
    best_clf = None
    best_meta: dict[str, Any] | None = None
    best_key = None
    for depth in TREE_DEPTH:
        for lr in TREE_LR:
            clf = HistGradientBoostingClassifier(
                max_depth=int(depth),
                learning_rate=float(lr),
                max_iter=120,
                random_state=int(seed),
            )
            clf.fit(x_tr, y_train)
            assign = predict_int(clf, x_va, val_ids)
            q, cost = assignment_quality_cost(matrix, val_ids, assign, cost_of)
            adv = val_quality_advantage(matrix, val_ids, assign, cost_of)
            key = score_tuple(adv, float(q.mean()), float(cost.mean()))
            if best_key is None or key > best_key:
                best_key = key
                best_clf = clf
                best_meta = {
                    "max_depth": int(depth),
                    "learning_rate": float(lr),
                    "max_iter": 120,
                    "degenerate": False,
                    "val_quality": float(q.mean()),
                    "val_cost": float(cost.mean()),
                    "val_quality_advantage": adv,
                    "selected_on": "validation",
                    "test_used": False,
                }
    assert best_clf is not None and best_meta is not None
    return best_clf, best_meta


def mix_from_assign(assign: Mapping[str, int], ids: Sequence[str]) -> dict[str, float]:
    n = len(ids)
    counts = {str(t): 0 for t in TIERS}
    for i in ids:
        counts[str(int(assign[i]))] += 1
    return {k: v / n for k, v in counts.items()}


def freeze_cost_matched_weights(matrix, ids: Sequence[str], assign: Mapping[str, int], cost_of) -> dict[str, Any]:
    cost = np.asarray([[cost_of(t, i) for t in TIERS] for i in ids], dtype=float)
    quality = np.asarray([[float(bool(matrix.correct[(t, i)])) for t in TIERS] for i in ids], dtype=float)
    q, c = assignment_quality_cost(matrix, ids, assign, cost_of)
    market = market_from_panel([str(t) for t in TIERS], cost, quality)
    mix = interpolate_at_cost(market, float(c.mean()), mode="equal")
    return {
        "weights": {str(k): float(v) for k, v in mix.weights.items()},
        "feasible": bool(mix.feasible),
        "kind": mix.kind,
        "note": mix.note,
        "frozen_from": "validation_hull_at_selected_router_val_cost",
        "val_router_cost": float(c.mean()),
        "val_router_quality": float(q.mean()),
        "test_used": False,
    }


def fit_train_preprocessor(
    train_texts: Sequence[str],
    train_dense: np.ndarray,
    *,
    seed: int,
) -> TrainOnlyFeatures:
    """TF-IDF + scaler. Callers must pass TRAIN texts only."""
    return TrainOnlyFeatures(
        train_texts,
        train_dense,
        dense_names=DENSE_NAMES,
        use_semantic=True,
        seed=int(seed),
        max_features=TFIDF_MAX_FEATURES,
    )


def refit_logistic(x_train, y_train: np.ndarray, *, C: float | None, seed: int):
    const = _maybe_constant(y_train)
    x_tr = as_dense(x_train)
    if const is not None or C is None:
        clf = const or ConstantClassifier(int(y_train[0]))
        return clf.fit(x_tr, y_train)
    clf = LogisticRegression(
        C=float(C),
        max_iter=800,
        solver="lbfgs",
        random_state=int(seed),
    )
    clf.fit(x_tr, y_train)
    return clf


def refit_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_depth: int | None,
    learning_rate: float | None,
    seed: int,
):
    const = _maybe_constant(y_train)
    x_tr = as_dense(x_train)
    if const is not None or max_depth is None or learning_rate is None:
        clf = const or ConstantClassifier(int(y_train[0]))
        return clf.fit(x_tr, y_train)
    clf = HistGradientBoostingClassifier(
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        max_iter=120,
        random_state=int(seed),
    )
    clf.fit(x_tr, y_train)
    return clf


def train(
    *,
    new_experiment: bool = False,
    seed: int = MODEL_SEED,
) -> dict[str, Any]:
    refuse_if_locked(new_experiment=new_experiment, action="retrain held-out routers")

    manifest = load_manifest()
    verify_manifest_hash(manifest)
    splits = split_ids(manifest)
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    matrix, records = load_panel()
    if set(train_ids) | set(val_ids) | set(test_ids) != set(matrix.item_ids):
        raise HeldoutLeakageError("manifest ids do not match the panel")
    if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids) or set(train_ids) & set(val_ids):
        raise HeldoutLeakageError("splits overlap")

    # Coverage only: test ids exist. Labels and texts are not used below.
    _ = len(test_ids)

    cost_of = cost_fn_usd(matrix)
    lex = _lexicons()
    train_text = texts_for(records, train_ids)
    val_text = texts_for(records, val_ids)
    x_tr_dense = dense_matrix(train_text, lex=lex)
    x_va_dense = dense_matrix(val_text, lex=lex)
    feats = fit_train_preprocessor(train_text, x_tr_dense, seed=seed)
    x_tr, _, _ = feats.transform(train_text, x_tr_dense)
    x_va, _, _ = feats.transform(val_text, x_va_dense)
    y_tr = oracle_labels(matrix, train_ids, cost_of)

    mean_cost, _ = mean_cost_quality(matrix, train_ids, cost_of)
    cheap, strong = cheap_strong_from_train(mean_cost)

    log_clf, log_meta = fit_logistic_grid(x_tr, y_tr, x_va, val_ids, matrix, cost_of, seed=seed)
    tree_clf, tree_meta = fit_tree_grid(
        x_tr_dense, y_tr, x_va_dense, val_ids, matrix, cost_of, seed=seed
    )
    thresh_meta = select_length_tau(
        val_ids, records, matrix, cost_of, cheap=cheap, strong=strong, lex=lex
    )

    log_assign = predict_int(log_clf, x_va, val_ids)
    tree_assign = predict_int(tree_clf, x_va_dense, val_ids)
    thresh_assign = length_assign(
        val_ids, records, tau=thresh_meta["tau"], cheap=cheap, strong=strong, lex=lex
    )
    heur_assign = heuristic_assign(val_ids, records, lex)
    heur_q, heur_c = assignment_quality_cost(matrix, val_ids, heur_assign, cost_of)
    heur_meta = {
        "kind": "frozen_production_classify_prompt_local_nlp",
        "fitted": False,
        "val_quality": float(heur_q.mean()),
        "val_cost": float(heur_c.mean()),
        "val_quality_advantage": val_quality_advantage(matrix, val_ids, heur_assign, cost_of),
        "selected_on": "validation",
        "test_used": False,
    }

    candidates = {
        "logistic": (log_assign, log_meta),
        "tree": (tree_assign, tree_meta),
        "threshold": (thresh_assign, thresh_meta),
        "ecologic_heuristic": (heur_assign, heur_meta),
    }
    winner = None
    winner_key = None
    for name, (assign, meta) in candidates.items():
        q, c = assignment_quality_cost(matrix, val_ids, assign, cost_of)
        key = score_tuple(meta.get("val_quality_advantage"), float(q.mean()), float(c.mean()))
        if winner_key is None or key > winner_key:
            winner_key = key
            winner = name
    assert winner is not None
    selected_assign = candidates[winner][0]
    cost_matched = freeze_cost_matched_weights(matrix, val_ids, selected_assign, cost_of)
    random_p = mix_from_assign(selected_assign, val_ids)

    config = {
        "protocol": "heldout_v1",
        "experiment_id": "heldout_v1",
        "historical_stage12_n364_not_used": True,
        "split_manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset"]["graded_sha256"],
        "seed": int(seed),
        "split_seed": int(manifest["seed"]),
        "n": dict(manifest["n"]),
        "selection_criterion": SELECTION_CRITERION,
        "selected_method": winner,
        "cheap_tier": int(cheap),
        "strong_tier": int(strong),
        "train_mean_usd_by_tier": {str(t): float(mean_cost[t]) for t in TIERS},
        "logistic": log_meta,
        "tree": tree_meta,
        "threshold": thresh_meta,
        "ecologic_heuristic": heur_meta,
        "cost_matched_static": cost_matched,
        "random_matched_mix": {
            "weights": random_p,
            "frozen_from": "selected_router_validation_mix",
            "test_used": False,
        },
        "test_ids_hash": hash_config({"test": split_ids(manifest)["test"]}),
        "test_metrics_inspected": False,
        "preprocessing": {
            "tfidf_fit_on": "train",
            "scaler_fit_on": "train",
            "dense_lexicons": "frozen_production (not redesigned on this panel)",
        },
    }
    config = sanitize(config)
    config["config_sha256"] = hash_config(config)
    write_result(FROZEN_CONFIG_REL, config, clobber=True)
    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train held-out routers; select on validation only.")
    p.add_argument("--new-experiment", action="store_true")
    p.add_argument("--seed", type=int, default=MODEL_SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = train(new_experiment=bool(args.new_experiment), seed=int(args.seed))
    except LockedExperimentError as exc:
        print(exc)
        return 1
    n = cfg["n"]
    print(f"TRAIN n\t{n['train']}")
    print(f"VAL n\t{n['val']}")
    print(f"TEST n\t{n['test']}")
    print(f"selected_method\t{cfg['selected_method']}")
    print(f"frozen_config\t{FROZEN_CONFIG_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
