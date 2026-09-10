"""Router ablations: which components actually change quality, cost, and regret.

Leakage protocol
----------------
- Preprocessing (TF-IDF, scalers) is fit on **train** only.
- Hyperparameters and thresholds are chosen on **validation** only.
- The test set is scored **once** per frozen configuration.
- Ablations are never selected using test metrics.
- Feature importances (when reported) are associative, not causal.

Production EcoLogic (`classify_prompt_local_nlp`) is a keyword/heuristic
decision list. Inspected families:

- lexical: closed lexicons (starters, code, risk, comparison, analysis words)
- complexity: multi-step phrases and multi-part questions
- length: word/character counts and short-query heuristics
- semantic: **not present** in production; TF-IDF is added only for learned routers
- confidence: **not present** in production or frozen logs
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from woais_experiments.accounting.costs import (
    TIERS,
    cost_fn_usd,
    evaluate_assignment,
    mean_cost_by_tier,
    naive_policy_cost,
    true_policy_cost,
)
from woais_experiments.deployment.router import production_classifier_namespace
from woais_experiments.frozen import ItemMatrix, load_stage12_matrix, open_frozen
from woais_experiments.paths import ROOT
from woais_experiments.routing.cost_matching import compare_router_to_static
from woais_experiments.routing.policies import oracle as oracle_assignment
from woais_experiments.routing.static_baselines import market_from_panel
from woais_experiments.statistics.paired_tests import apply_bh, paired_comparison

PRODUCTION_FAMILIES = ("lexical", "complexity", "length")
LEARNED_FAMILIES = ("lexical", "complexity", "length", "semantic")
ALL_FAMILIES = ("lexical", "complexity", "length", "semantic", "confidence")
REFERENCE = "full"
SPLIT_SEED = 20260909
MODEL_SEED = 20260909
DEFAULT_N_BOOT = 2000
DEFAULT_N_PERM = 199
DEFAULT_LEVEL = 0.95
TRAIN_FRAC = 0.50
VAL_FRAC = 0.25
# test gets the remainder (~0.25)

MULTI_STEP_PHRASES = (
    "step by step",
    "pros and cons",
    "advantages and disadvantages",
    "first then",
    "both",
    "each",
)

LEXICAL_NAMES = (
    "simple_starter",
    "has_code_action",
    "has_prog_lang",
    "has_code_debug",
    "has_code_noun",
    "code_action_and_lang",
    "debug_and_lang_or_noun",
    "noun_and_lang",
    "has_risk_domain",
    "has_risk_action",
    "risk_and_action",
    "has_comparison",
    "n_comparison_hits",
    "n_analysis_hits",
)
COMPLEXITY_NAMES = (
    "has_analysis",
    "has_multistep",
    "multi_question",
    "n_sentences",
)
LENGTH_NAMES = (
    "n_words",
    "n_chars",
    "log_n_words",
    "mean_word_len",
    "n_question_marks",
)

DENSE_NAMES = LEXICAL_NAMES + COMPLEXITY_NAMES + LENGTH_NAMES
FAMILY_OF_DENSE = {
    **{n: "lexical" for n in LEXICAL_NAMES},
    **{n: "complexity" for n in COMPLEXITY_NAMES},
    **{n: "length" for n in LENGTH_NAMES},
}


class AblationLeakageError(RuntimeError):
    """Train/val/test protocol was violated."""


class TestQueryMismatchError(RuntimeError):
    """Ablation variants did not score the same test queries."""


def inspect_feature_families() -> dict[str, Any]:
    """Inventory of families in production vs learned routers."""
    return {
        "production_router": "backend/main.py:classify_prompt_local_nlp",
        "production_kind": "keyword_heuristic_decision_list",
        "families": {
            "lexical": {
                "present_in_production": True,
                "signals": [
                    "SIMPLE_STARTERS",
                    "PROGRAMMING_LANGS",
                    "CODE_ACTIONS",
                    "CODE_DEBUG",
                    "CODE_NOUNS",
                    "HIGH_RISK_DOMAINS",
                    "HIGH_RISK_ACTIONS",
                    "COMPARISON_WORDS",
                    "ANALYSIS_WORDS",
                ],
            },
            "complexity": {
                "present_in_production": True,
                "signals": ["multi_step_phrases", "analysis_words", "multi_part_questions"],
            },
            "length": {
                "present_in_production": True,
                "signals": ["word_count", "question_marks", "short_query_heuristic"],
            },
            "semantic": {
                "present_in_production": False,
                "used_in_learned_routers": True,
                "implementation": "TfidfVectorizer fit on train text only",
                "note": "MiniLM embeddings exist in frozen router_v2/ but are not imported here.",
            },
            "confidence": {
                "present_in_production": False,
                "used_in_learned_routers": False,
                "note": "No per-query confidence is stored in frozen logs.",
            },
        },
        "importance_note": (
            "Feature importance is reported only for fitted logistic/tree models "
            "and is associative, not causal."
        ),
    }


def load_item_texts() -> dict[str, dict[str, str]]:
    """Read frozen benchmark items (raw_query is the production routing input)."""
    path = ROOT / "raw_results" / "benchmark_items.json"
    with open_frozen(path) as fh:
        payload = json.load(fh)
    out = {}
    for it in payload["items"]:
        qid = str(it["item_id"])
        out[qid] = {
            "raw_query": str(it.get("raw_query") or it.get("prompt") or ""),
            "prompt": str(it.get("prompt") or ""),
            "benchmark": str(it.get("benchmark") or "unknown"),
        }
    return out


def subset_matrix(matrix: ItemMatrix, ids: Sequence[str]) -> ItemMatrix:
    keep = [str(i) for i in ids]
    def _take(store: dict) -> dict:
        return {(t, i): store[(t, i)] for t in TIERS for i in keep}
    return ItemMatrix(
        item_ids=keep,
        bench_of={i: matrix.bench_of[i] for i in keep},
        correct=_take(matrix.correct),
        tokens=_take(matrix.tokens),
        prompt_tokens=_take(matrix.prompt_tokens),
        completion_tokens=_take(matrix.completion_tokens),
        usd=_take(matrix.usd),
        latency_s=_take(matrix.latency_s),
        model_of=dict(matrix.model_of),
        n_dropped_incomplete=0,
    )


def stratified_splits(
    item_ids: Sequence[str],
    bench_of: Mapping[str, str],
    *,
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, list[str]]:
    """Disjoint train/val/test, stratified by benchmark. No test in fitting."""
    rng = np.random.default_rng(int(seed))
    by_b: dict[str, list[str]] = {}
    for qid in item_ids:
        by_b.setdefault(str(bench_of.get(qid, "unknown")), []).append(str(qid))
    train, val, test = [], [], []
    for bench in sorted(by_b):
        ids = list(by_b[bench])
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train + n_val >= n and n >= 3:
            n_val = max(1, n - n_train - 1)
        n_train = min(n_train, n - 2) if n >= 3 else max(1, n // 2)
        n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else 0
        train.extend(ids[:n_train])
        val.extend(ids[n_train:n_train + n_val])
        test.extend(ids[n_train + n_val:])
    if not val and train:
        val.append(train.pop())
    if not test and train:
        test.append(train.pop())
    split = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    assert_disjoint_splits(split)
    if set(split["train"]) | set(split["val"]) | set(split["test"]) != set(map(str, item_ids)):
        raise AblationLeakageError("split does not cover every item exactly once")
    return split


def assert_disjoint_splits(split: Mapping[str, Sequence[str]]) -> None:
    t, v, te = set(split["train"]), set(split["val"]), set(split["test"])
    if t & v or t & te or v & te:
        raise AblationLeakageError("train/val/test are not disjoint")
    if not te:
        raise AblationLeakageError("empty test split")


def _lexicons() -> dict[str, Any]:
    ns = production_classifier_namespace()
    return {
        "tokenize": ns["simple_tokenize"],
        "simple_starters": set(ns["SIMPLE_STARTERS"]),
        "langs": set(ns["PROGRAMMING_LANGS"]),
        "code_actions": set(ns["CODE_ACTIONS"]),
        "code_debug": set(ns["CODE_DEBUG"]),
        "code_nouns": set(ns["CODE_NOUNS"]),
        "risk_domains": set(ns["HIGH_RISK_DOMAINS"]),
        "risk_actions": set(ns["HIGH_RISK_ACTIONS"]),
        "comparison": set(ns["COMPARISON_WORDS"]),
        "analysis": set(ns["ANALYSIS_WORDS"]),
        "classify": ns["classify_prompt_local_nlp"],
    }


def dense_features(text: str, *, lex: dict[str, Any] | None = None) -> dict[str, float]:
    lex = lex or _lexicons()
    prompt = text or ""
    prompt_lower = prompt.lower()
    tokens = lex["tokenize"](prompt)
    words_set = set(tokens)
    word_count = len(tokens)
    first_three = " ".join(tokens[:3])
    qmarks = prompt.count("?")
    row = {
        "simple_starter": float(any(prompt_lower.startswith(s) for s in lex["simple_starters"])),
        "has_code_action": float(bool(lex["code_actions"] & words_set)),
        "has_prog_lang": float(bool(lex["langs"] & words_set)),
        "has_code_debug": float(bool(lex["code_debug"] & words_set)),
        "has_code_noun": float(bool(lex["code_nouns"] & words_set)),
        "has_risk_domain": float(bool(lex["risk_domains"] & words_set)),
        "has_risk_action": float(bool(lex["risk_actions"] & words_set)),
        "has_comparison": float(bool(lex["comparison"] & words_set)),
        "n_comparison_hits": float(len(lex["comparison"] & words_set)),
        "n_analysis_hits": float(len(lex["analysis"] & words_set)),
        "has_analysis": float(bool(lex["analysis"] & words_set)),
        "has_multistep": float(any(p in prompt_lower for p in MULTI_STEP_PHRASES)),
        "multi_question": float(word_count > 25 and qmarks >= 2),
        "n_sentences": float(max(prompt.count(".") + prompt.count("!") + qmarks, 1)),
        "n_words": float(word_count),
        "n_chars": float(len(prompt)),
        "log_n_words": float(math.log1p(word_count)),
        "mean_word_len": float(sum(len(w) for w in tokens) / word_count) if word_count else 0.0,
        "n_question_marks": float(qmarks),
    }
    row["code_action_and_lang"] = row["has_code_action"] * row["has_prog_lang"]
    row["debug_and_lang_or_noun"] = row["has_code_debug"] * max(row["has_prog_lang"], row["has_code_noun"])
    row["noun_and_lang"] = row["has_code_noun"] * row["has_prog_lang"]
    row["risk_and_action"] = row["has_risk_domain"] * row["has_risk_action"]
    row["_first_three_what"] = float(any(s in first_three for s in ("what is", "what are")))
    return row


def gated_ecologic_tier(
    text: str,
    *,
    families: Sequence[str] = PRODUCTION_FAMILIES,
    lex: dict[str, Any] | None = None,
) -> int:
    """Production decision list with named families gated off.

    Disabled families never fire. The default remains Tier 1. When every
    production family is enabled this matches ``classify_prompt_local_nlp``.
    """
    enabled = set(families)
    lex = lex or _lexicons()
    f = dense_features(text, lex=lex)
    use_lex = "lexical" in enabled
    use_cx = "complexity" in enabled
    use_len = "length" in enabled

    if use_lex and f["simple_starter"]:
        return 1
    if use_lex and f["code_action_and_lang"]:
        return 3
    if use_lex and f["debug_and_lang_or_noun"]:
        return 3
    if use_lex and f["noun_and_lang"]:
        return 3
    if use_lex and f["risk_and_action"]:
        return 3
    if use_lex and f["has_comparison"] and not f["_first_three_what"]:
        return 2
    if use_lex and f["has_analysis"]:
        return 2
    if use_cx and f["has_multistep"]:
        return 2
    if use_cx and use_len and f["n_words"] > 25 and f["n_question_marks"] >= 2:
        return 2
    if use_len and f["n_words"] <= 10 and f["n_question_marks"] <= 1:
        return 1
    return 1


def assign_gated(
    ids: Sequence[str],
    texts: Mapping[str, str],
    families: Sequence[str],
    *,
    lex: dict[str, Any] | None = None,
) -> dict[str, int]:
    lex = lex or _lexicons()
    return {str(i): gated_ecologic_tier(texts[str(i)], families=families, lex=lex) for i in ids}


def cheap_strong_tiers(matrix: ItemMatrix, ids: Sequence[str], cost_of) -> tuple[int, int]:
    """Lowest / highest mean cost on ``ids`` (train or val). Never uses test."""
    n = len(ids)
    means = {t: sum(cost_of(t, i) for i in ids) / n for t in TIERS}
    cheap = min(TIERS, key=lambda t: (means[t], t))
    strong = max(TIERS, key=lambda t: (means[t], -t))
    return int(cheap), int(strong)


def mix_of(assign: Mapping[str, int], ids: Sequence[str]) -> dict[int, float]:
    n = len(ids)
    return {t: sum(1 for i in ids if int(assign[i]) == t) / n for t in TIERS}


def random_matched_assignment(
    ids: Sequence[str],
    mix: Mapping[int, float],
    *,
    seed: int,
) -> dict[str, int]:
    probs = np.array([float(mix[t]) for t in TIERS], dtype=float)
    s = float(probs.sum())
    if s <= 0:
        probs = np.ones(len(TIERS)) / len(TIERS)
    else:
        probs = probs / s
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(list(TIERS), size=len(ids), p=probs)
    return {str(i): int(t) for i, t in zip(ids, draws)}


def static_quota_assignment(
    ids: Sequence[str],
    mix: Mapping[int, float],
    *,
    seed: int,
) -> dict[str, int]:
    """Query-independent mix: seeded permutation + largest-remainder quotas."""
    n = len(ids)
    raw = [float(mix[t]) * n for t in TIERS]
    floors = [int(math.floor(x)) for x in raw]
    leftover = n - sum(floors)
    frac = sorted(enumerate([x - f for x, f in zip(raw, floors)]), key=lambda z: -z[1])
    counts = list(floors)
    for k in range(leftover):
        counts[frac[k % len(frac)][0]] += 1
    rng = np.random.default_rng(int(seed))
    order = list(ids)
    rng.shuffle(order)
    assign: dict[str, int] = {}
    cursor = 0
    for t, c in zip(TIERS, counts):
        for qid in order[cursor:cursor + c]:
            assign[str(qid)] = int(t)
        cursor += c
    for qid in ids:
        assign.setdefault(str(qid), int(TIERS[0]))
    return assign


def always_assignment(ids: Sequence[str], tier: int) -> dict[str, int]:
    return {str(i): int(tier) for i in ids}


def _stack_dense(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(r[n]) for n in names] for r in rows], dtype=float)


def _family_dense_names(families: Sequence[str]) -> list[str]:
    keep = set(families)
    return [n for n in DENSE_NAMES if FAMILY_OF_DENSE[n] in keep]


class TrainOnlyFeatures:
    """TF-IDF + scaler fit on train texts/dense columns only."""

    def __init__(
        self,
        train_texts: Sequence[str],
        train_dense: np.ndarray,
        *,
        dense_names: Sequence[str],
        use_semantic: bool,
        seed: int = MODEL_SEED,
        max_features: int = 512,
    ) -> None:
        self.dense_names = list(dense_names)
        self.use_semantic = bool(use_semantic)
        self.scaler = StandardScaler()
        if train_dense.size:
            self.scaler.fit(train_dense)
        else:
            self.scaler = None
        self.tfidf: TfidfVectorizer | None = None
        if use_semantic:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=int(max_features),
                lowercase=True,
            )
            self.tfidf.fit(list(train_texts))
        self.seed = int(seed)

    def transform(self, texts: Sequence[str], dense: np.ndarray) -> Any:
        parts = []
        names: list[str] = []
        families: list[str] = []
        if self.dense_names:
            x = self.scaler.transform(dense) if self.scaler is not None else dense
            parts.append(csr_matrix(np.asarray(x, dtype=float)))
            names.extend(self.dense_names)
            families.extend(FAMILY_OF_DENSE[n] for n in self.dense_names)
        if self.tfidf is not None:
            xt = self.tfidf.transform(list(texts))
            parts.append(xt)
            vocab = list(self.tfidf.get_feature_names_out())
            names.extend(f"tfidf:{t}" for t in vocab)
            families.extend("semantic" for _ in vocab)
        if not parts:
            return csr_matrix((len(texts), 0)), [], []
        return hstack(parts).tocsr(), names, families


def _val_accuracy(
    pred: Mapping[str, int],
    matrix: ItemMatrix,
    ids: Sequence[str],
) -> float:
    if not ids:
        return float("nan")
    return sum(bool(matrix.correct[(int(pred[i]), i)]) for i in ids) / len(ids)


def fit_logistic(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    *,
    seed: int,
    grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[LogisticRegression, dict[str, Any]]:
    best = None
    best_acc = -1.0
    best_c = None
    if len(np.unique(y_train)) < 2:
        clf = LogisticRegression(max_iter=400, random_state=int(seed))
        clf.classes_ = np.unique(y_train)
        # Constant predictor: sklearn needs fit; duplicate a dummy class by skipping.
        const = int(y_train[0])
        class _Const:
            classes_ = np.array([const])
            def predict(self, X):
                return np.full(X.shape[0], const, dtype=int)
            coef_ = np.zeros((1, x_train.shape[1]))
        return _Const(), {"C": None, "val_accuracy": float("nan"), "constant": const}
    for c in grid:
        clf = LogisticRegression(
            C=float(c),
            max_iter=400,
            random_state=int(seed),
            solver="lbfgs",
        )
        clf.fit(x_train, y_train)
        acc = float(np.mean(clf.predict(x_val) == y_val)) if x_val.shape[0] else float("nan")
        if acc > best_acc:
            best_acc = acc
            best = clf
            best_c = float(c)
    assert best is not None
    return best, {"C": best_c, "val_accuracy": best_acc, "grid": list(grid)}


def fit_tree(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    *,
    seed: int,
    depths: Sequence[int] = (1, 2, 3),
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    best = None
    best_acc = -1.0
    best_d = None
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=int(d), random_state=int(seed), min_samples_leaf=2)
        clf.fit(x_train, y_train)
        acc = float(np.mean(clf.predict(x_val) == y_val)) if x_val.shape[0] else float("nan")
        if acc > best_acc:
            best_acc = acc
            best = clf
            best_d = int(d)
    assert best is not None
    return best, {"max_depth": best_d, "val_accuracy": best_acc, "grid": list(depths)}


def select_length_threshold(
    val_length: np.ndarray,
    val_oracle: np.ndarray,
    *,
    cheap: int,
    strong: int,
) -> dict[str, Any]:
    """Two-way length router: length >= tau → strong else cheap. Tau from val."""
    if val_length.size == 0:
        return {"tau": 0.0, "val_accuracy": float("nan")}
    candidates = np.unique(np.quantile(val_length, np.linspace(0, 1, 21)))
    best_tau = float(candidates[0])
    best_acc = -1.0
    y = np.asarray(val_oracle)
    for tau in candidates:
        pred = np.where(val_length >= float(tau), strong, cheap)
        acc = float(np.mean(pred == y))
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    return {"tau": best_tau, "val_accuracy": best_acc, "cheap": int(cheap), "strong": int(strong)}


def apply_length_threshold(
    ids: Sequence[str],
    lengths: Mapping[str, float],
    *,
    tau: float,
    cheap: int,
    strong: int,
) -> dict[str, int]:
    return {str(i): (int(strong) if float(lengths[str(i)]) >= float(tau) else int(cheap)) for i in ids}


def select_strongest_family(
    texts: Mapping[str, str],
    val_ids: Sequence[str],
    matrix: ItemMatrix,
    *,
    lex: dict[str, Any],
) -> dict[str, Any]:
    """Pick a single production family using validation accuracy only."""
    scores = {}
    for fam in PRODUCTION_FAMILIES:
        pred = assign_gated(val_ids, texts, [fam], lex=lex)
        scores[fam] = _val_accuracy(pred, matrix, val_ids)
    # Tie-break: name, never test.
    best = max(PRODUCTION_FAMILIES, key=lambda f: (scores[f], f))
    return {
        "family": best,
        "val_accuracy": scores[best],
        "val_scores": scores,
        "selected_on": "validation",
        "criterion": "accuracy",
        "test_used": False,
    }


def _predict_model(clf, x, ids: Sequence[str]) -> dict[str, int]:
    pred = clf.predict(x)
    return {str(i): int(t) for i, t in zip(ids, pred)}


def _importance_logistic(clf, names: Sequence[str], families: Sequence[str]) -> dict[str, Any] | None:
    coef = getattr(clf, "coef_", None)
    if coef is None or not names:
        return None
    mag = np.mean(np.abs(np.asarray(coef, dtype=float)), axis=0)
    by: dict[str, float] = {}
    for fam in ALL_FAMILIES:
        idx = [i for i, f in enumerate(families) if f == fam]
        by[fam] = float(mag[idx].sum()) if idx else 0.0
    top = sorted(zip(names, mag.tolist(), families), key=lambda z: -z[1])[:15]
    return {
        "kind": "mean_abs_logistic_coefficient",
        "causal": False,
        "note": "Associative coefficient magnitude, not a causal effect.",
        "by_family": by,
        "top_features": [{"name": n, "magnitude": m, "family": f} for n, m, f in top],
    }


def _importance_tree(clf, names: Sequence[str], families: Sequence[str]) -> dict[str, Any] | None:
    imp = getattr(clf, "feature_importances_", None)
    if imp is None or not names:
        return None
    mag = np.asarray(imp, dtype=float)
    by: dict[str, float] = {}
    for fam in ALL_FAMILIES:
        idx = [i for i, f in enumerate(families) if f == fam]
        by[fam] = float(mag[idx].sum()) if idx else 0.0
    top = sorted(zip(names, mag.tolist(), families), key=lambda z: -z[1])[:15]
    return {
        "kind": "gini_decrease",
        "causal": False,
        "note": "Impurity decrease is not a causal effect.",
        "by_family": by,
        "top_features": [{"name": n, "magnitude": m, "family": f} for n, m, f in top],
    }


def cost_quality_panels(matrix: ItemMatrix, ids: Sequence[str], cost_of) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray([[cost_of(t, i) for t in TIERS] for i in ids], dtype=float)
    quality = np.asarray(
        [[float(bool(matrix.correct[(t, i)])) for t in TIERS] for i in ids],
        dtype=float,
    )
    return cost, quality


def evaluate_variant(
    name: str,
    assign: Mapping[str, int],
    matrix: ItemMatrix,
    ids: Sequence[str],
    cost_of,
    *,
    oracle_assign: Mapping[str, int],
    cheap: int,
    strong: int,
) -> dict[str, Any]:
    sub = subset_matrix(matrix, ids)
    stats = evaluate_assignment(sub, dict(assign), cost_of, name=name)
    mix = mix_of(assign, ids)
    means = mean_cost_by_tier(sub, cost_of)
    naive = naive_policy_cost(mix, means)
    realized = true_policy_cost(assign, cost_of, ids)
    cost, quality = cost_quality_panels(sub, ids, cost_of)
    named = {i: str(assign[i]) for i in ids}
    market = market_from_panel([str(t) for t in TIERS], cost, quality)
    rq = float(stats["accuracy"])
    rc = float(stats["cost_per_item"])
    vs_static = compare_router_to_static(market, rq, rc)
    o_stats = evaluate_assignment(sub, dict(oracle_assign), cost_of, name="oracle")
    per_q_quality = [float(bool(sub.correct[(assign[i], i)])) for i in ids]
    per_q_cost = [float(cost_of(assign[i], i)) for i in ids]
    per_q_regret = [float(cost_of(assign[i], i) - cost_of(oracle_assign[i], i)) for i in ids]
    return {
        "name": name,
        "n": len(ids),
        "quality": float(stats["accuracy"]),
        "realized_cost": float(realized),
        "naive_cost": float(naive),
        "strong_model_routing_rate": float(mix[strong]),
        "cheap_model_routing_rate": float(mix[cheap]),
        "tier_mix": {str(t): mix[t] for t in TIERS},
        "quality_at_matched_cost": vs_static.get("static_quality_at_same_cost"),
        "cost_at_matched_quality": vs_static.get("static_cost_at_same_quality"),
        "router_vs_static_advantage": vs_static.get("quality_advantage"),
        "oracle_gap_quality": float(o_stats["accuracy"] - stats["accuracy"]),
        "oracle_gap_cost": float(realized - o_stats["cost_per_item"]),
        "latency_mean_s": stats.get("latency_mean_s"),
        "latency_p50_s": stats.get("latency_p50_s"),
        "vs_static": vs_static,
        "eval": stats,
        "query_quality": dict(zip(ids, per_q_quality)),
        "query_cost": dict(zip(ids, per_q_cost)),
        "query_regret": dict(zip(ids, per_q_regret)),
        "assignment": {str(i): int(assign[i]) for i in ids},
        "named_assignment": named,
    }


def paired_vs_full(
    variants: Mapping[str, dict[str, Any]],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    full = variants[REFERENCE]
    names = [n for n in variants if n != REFERENCE]
    blocks = {"quality": [], "cost": [], "regret": []}
    reports: dict[str, Any] = {}
    for name in names:
        row = {}
        for endpoint, key in (("quality", "query_quality"), ("cost", "query_cost"), ("regret", "query_regret")):
            cmp_ = paired_comparison(
                variants[name][key],
                full[key],
                name_a=name,
                name_b=REFERENCE,
                name=f"{name}-full:{endpoint}",
                n_boot=n_boot,
                n_perm=n_perm,
                seed=seed,
                level=level,
            )
            mean = cmp_["mean_paired_difference"]
            row[endpoint] = {
                "delta": mean.get("estimate"),
                "ci_lo": mean.get("ci_lo"),
                "ci_hi": mean.get("ci_hi"),
                "p_raw": mean.get("p_raw"),
                "p_adjusted": None,
                "effect_size": {
                    "cohens_dz": cmp_["cohens_dz"].get("estimate"),
                    "cliffs_delta": cmp_["cliffs_delta"].get("estimate"),
                },
                "n": cmp_["n"],
                "comparison": cmp_,
            }
            blocks[endpoint].append(row[endpoint])
        reports[name] = row
    for endpoint, estimates in blocks.items():
        adjusted = apply_bh(estimates)
        for name, est in zip(names, adjusted):
            reports[name][endpoint]["p_adjusted"] = est.get("p_adjusted")
            reports[name][endpoint]["bh_family"] = f"{endpoint}_vs_{REFERENCE}"
    return reports


def assert_same_test_queries(variants: Mapping[str, Mapping[str, Any]], test_ids: Sequence[str]) -> None:
    expected = [str(i) for i in test_ids]
    for name, payload in variants.items():
        got = sorted(payload["assignment"])
        if got != sorted(expected):
            raise TestQueryMismatchError(
                f"{name} scored {len(got)} queries; expected {len(expected)} identical test ids"
            )
        if set(got) != set(expected):
            raise TestQueryMismatchError(f"{name} test query set differs from the frozen test split")


def _oracle_labels(matrix: ItemMatrix, ids: Sequence[str], cost_of) -> dict[str, int]:
    full = oracle_assignment(matrix, cost_of)
    return {str(i): int(full[i]) for i in ids}


def run_ablation_study(
    *,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = SPLIT_SEED,
    model_seed: int = MODEL_SEED,
    matrix: ItemMatrix | None = None,
    texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fit on train, tune on val, evaluate each frozen config once on test."""
    matrix = matrix or load_stage12_matrix()
    item_meta = load_item_texts()
    if texts is None:
        texts = {qid: item_meta[qid]["raw_query"] for qid in matrix.item_ids if qid in item_meta}
    missing = [i for i in matrix.item_ids if i not in texts]
    if missing:
        raise AblationLeakageError(f"missing raw_query for {len(missing)} items")
    lex = _lexicons()
    cost_of = cost_fn_usd(matrix)
    split = stratified_splits(matrix.item_ids, matrix.bench_of, seed=seed)
    train_ids, val_ids, test_ids = split["train"], split["val"], split["test"]
    cheap, strong = cheap_strong_tiers(matrix, train_ids, cost_of)
    oracle_test = _oracle_labels(matrix, test_ids, cost_of)
    oracle_train = _oracle_labels(matrix, train_ids, cost_of)
    oracle_val = _oracle_labels(matrix, val_ids, cost_of)

    # --- production-gated (no label fitting) ---
    full_all = assign_gated(matrix.item_ids, texts, PRODUCTION_FAMILIES, lex=lex)
    mismatches = [
        qid for qid in matrix.item_ids
        if int(full_all[qid]) != int(lex["classify"](texts[qid]).recommended_tier)
    ]
    if mismatches:
        raise AblationLeakageError(
            f"gated full router disagrees with production on {len(mismatches)} items"
        )

    strongest = select_strongest_family(texts, val_ids, matrix, lex=lex)
    val_full = {i: full_all[i] for i in val_ids}
    val_mix = mix_of(val_full, val_ids)

    # --- learned features: train-only fit ---
    def dense_block(ids: Sequence[str], names: Sequence[str]) -> np.ndarray:
        rows = [dense_features(texts[i], lex=lex) for i in ids]
        if not names:
            return np.zeros((len(ids), 0), dtype=float)
        return _stack_dense(rows, names)

    all_dense_names = list(DENSE_NAMES)
    feats_all = TrainOnlyFeatures(
        [texts[i] for i in train_ids],
        dense_block(train_ids, all_dense_names),
        dense_names=all_dense_names,
        use_semantic=True,
        seed=model_seed,
    )
    feats_no_sem = TrainOnlyFeatures(
        [texts[i] for i in train_ids],
        dense_block(train_ids, all_dense_names),
        dense_names=all_dense_names,
        use_semantic=False,
        seed=model_seed,
    )

    def design(feat: TrainOnlyFeatures, ids: Sequence[str], names: Sequence[str] | None = None):
        names = list(names) if names is not None else list(feat.dense_names)
        # Rebuild a transformer if dense subset differs — keep train-only scaler
        # by slicing columns after transform of the full dense block.
        x, fnames, ffam = feat.transform([texts[i] for i in ids], dense_block(ids, feat.dense_names))
        if names != list(feat.dense_names):
            keep = [i for i, n in enumerate(fnames) if n in set(names) or str(n).startswith("tfidf:")]
            if names != list(feat.dense_names) and not feat.use_semantic:
                keep = [i for i, n in enumerate(fnames) if n in set(names)]
            x = x[:, keep]
            fnames = [fnames[i] for i in keep]
            ffam = [ffam[i] for i in keep]
        return x, fnames, ffam

    y_tr = np.array([oracle_train[i] for i in train_ids], dtype=int)
    y_va = np.array([oracle_val[i] for i in val_ids], dtype=int)
    xtr, names_all, fam_all = design(feats_all, train_ids)
    xva, _, _ = design(feats_all, val_ids)
    xte, _, _ = design(feats_all, test_ids)
    log_clf, log_meta = fit_logistic(xtr, y_tr, xva, y_va, seed=model_seed)
    tree_clf, tree_meta = fit_tree(xtr, y_tr, xva, y_va, seed=model_seed)

    lengths = {i: dense_features(texts[i], lex=lex)["log_n_words"] for i in matrix.item_ids}
    val_len = np.array([lengths[i] for i in val_ids], dtype=float)
    thresh = select_length_threshold(
        val_len, np.array([oracle_val[i] for i in val_ids]), cheap=cheap, strong=strong
    )

    variants_assign: dict[str, dict[str, int]] = {
        "full": {i: full_all[i] for i in test_ids},
        "no_semantic": {i: full_all[i] for i in test_ids},  # family absent in production
        "no_complexity": assign_gated(test_ids, texts, ("lexical", "length"), lex=lex),
        "no_length": assign_gated(test_ids, texts, ("lexical", "complexity"), lex=lex),
        "no_lexical": assign_gated(test_ids, texts, ("complexity", "length"), lex=lex),
        "no_confidence": {i: full_all[i] for i in test_ids},  # family absent
        "single_strongest_family": assign_gated(test_ids, texts, (strongest["family"],), lex=lex),
        "threshold_only": apply_length_threshold(
            test_ids, lengths, tau=thresh["tau"], cheap=cheap, strong=strong
        ),
        "logistic": _predict_model(log_clf, xte, test_ids),
        "tree": _predict_model(tree_clf, xte, test_ids),
        "random_matched_rate": random_matched_assignment(test_ids, val_mix, seed=model_seed),
        "static_mixture_matched_rate": static_quota_assignment(test_ids, val_mix, seed=model_seed + 1),
        "always_cheap": always_assignment(test_ids, cheap),
        "always_strong": always_assignment(test_ids, strong),
        "oracle": oracle_test,
    }

    notes = {
        "full": "Production EcoLogic decision list, all production families on.",
        "no_semantic": "Semantic features are not present in production; identical to full.",
        "no_complexity": "Production list with complexity rules gated off.",
        "no_length": "Production list with length heuristics gated off.",
        "no_lexical": "Production list with closed lexicons gated off.",
        "no_confidence": "Confidence features are not present; identical to full.",
        "single_strongest_family": (
            f"Production list using only '{strongest['family']}', selected on validation accuracy."
        ),
        "threshold_only": "Length threshold (log n_words) cheap vs strong; tau from validation.",
        "logistic": "Multinomial logistic on train-only TF-IDF + dense families; C from validation. Target = train oracle.",
        "tree": "Shallow decision tree on the same train-only features; depth from validation. Target = train oracle.",
        "random_matched_rate": "Query-independent multinomial with full-router mix estimated on validation.",
        "static_mixture_matched_rate": "Query-independent quota mix matching validation full-router rates.",
        "always_cheap": "Constant cheapest tier by train mean USD.",
        "always_strong": "Constant highest-cost tier by train mean USD.",
        "oracle": "Per-query cheapest correct tier on the test labels (upper bound).",
    }
    degenerate = {"no_semantic", "no_confidence"}

    scored = {}
    for name, assign in variants_assign.items():
        scored[name] = evaluate_variant(
            name, assign, matrix, test_ids, cost_of,
            oracle_assign=oracle_test, cheap=cheap, strong=strong,
        )
        scored[name]["note"] = notes[name]
        scored[name]["degenerate_identical_to_full"] = name in degenerate
    assert_same_test_queries(scored, test_ids)

    comparisons = paired_vs_full(scored, n_boot=n_boot, n_perm=n_perm, seed=seed)

    importance = {
        "logistic": _importance_logistic(log_clf, names_all, fam_all),
        "tree": _importance_tree(tree_clf, names_all, fam_all),
        "note": "Importances are associative, not causal. Not reported for non-fitted variants.",
    }

    configs = {
        "protocol": {
            "preprocessing": "train",
            "hyperparameters": "validation",
            "thresholds": "validation",
            "family_selection": "validation",
            "test": "once_per_frozen_configuration",
            "select_ablation_on_test": False,
        },
        "feature_families": inspect_feature_families(),
        "split": {
            "seed": int(seed),
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "cheap_tier": cheap,
        "strong_tier": strong,
        "cheap_strong_source": "train_mean_usd",
        "matched_rate_source": "full_router_validation_mix",
        "validation_full_mix": {str(k): v for k, v in val_mix.items()},
        "strongest_family": strongest,
        "logistic": log_meta,
        "tree": tree_meta,
        "threshold_only": thresh,
        "tfidf": {
            "fitted_on": "train",
            "max_features": 512,
            "ngram_range": [1, 2],
            "min_df": 2,
            "n_terms": 0 if feats_all.tfidf is None else len(feats_all.tfidf.get_feature_names_out()),
        },
        "importance": importance,
        "notes": notes,
        "degenerate_variants": sorted(degenerate),
        "reference": REFERENCE,
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "model_seed": int(model_seed),
        "gated_full_matches_production": True,
        "test_queries_identical": True,
        "variants": {
            name: {
                "note": notes[name],
                "degenerate_identical_to_full": name in degenerate,
            }
            for name in variants_assign
        },
    }
    return {
        "configs": configs,
        "variants": scored,
        "comparisons": comparisons,
        "split": split,
        "test_ids": test_ids,
        "texts": texts,
        "oracle_test": oracle_test,
        "matrix": matrix,
        "cost_of": cost_of,
    }


def summary_rows(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, v in study["variants"].items():
        cmp_ = study["comparisons"].get(name) or {}
        def _ep(endpoint: str, field: str):
            if name == REFERENCE:
                if field == "delta":
                    return 0.0
                if field in {"ci_lo", "ci_hi"}:
                    return 0.0
                if field == "p_raw":
                    return 1.0
                return None
            block = (cmp_.get(endpoint) or {})
            return block.get(field)
        rows.append({
            "variant": name,
            "n": v["n"],
            "quality": v["quality"],
            "realized_cost": v["realized_cost"],
            "naive_cost": v["naive_cost"],
            "strong_model_routing_rate": v["strong_model_routing_rate"],
            "quality_at_matched_cost": v["quality_at_matched_cost"],
            "cost_at_matched_quality": v["cost_at_matched_quality"],
            "router_vs_static_advantage": v["router_vs_static_advantage"],
            "oracle_gap": v["oracle_gap_quality"],
            "oracle_gap_cost": v["oracle_gap_cost"],
            "latency_mean_s": v["latency_mean_s"],
            "delta_quality": _ep("quality", "delta"),
            "delta_cost": _ep("cost", "delta"),
            "delta_regret": _ep("regret", "delta"),
            "delta_quality_ci_lo": _ep("quality", "ci_lo"),
            "delta_quality_ci_hi": _ep("quality", "ci_hi"),
            "delta_cost_ci_lo": _ep("cost", "ci_lo"),
            "delta_cost_ci_hi": _ep("cost", "ci_hi"),
            "delta_regret_ci_lo": _ep("regret", "ci_lo"),
            "delta_regret_ci_hi": _ep("regret", "ci_hi"),
            "p_raw_quality": _ep("quality", "p_raw"),
            "p_adjusted_quality": _ep("quality", "p_adjusted"),
            "p_raw_cost": _ep("cost", "p_raw"),
            "p_adjusted_cost": _ep("cost", "p_adjusted"),
            "p_raw_regret": _ep("regret", "p_raw"),
            "p_adjusted_regret": _ep("regret", "p_adjusted"),
            "effect_size_quality_dz": (cmp_.get("quality") or {}).get("effect_size", {}).get("cohens_dz") if name != REFERENCE else 0.0,
            "effect_size_quality_cliff": (cmp_.get("quality") or {}).get("effect_size", {}).get("cliffs_delta") if name != REFERENCE else 0.0,
            "degenerate_identical_to_full": v.get("degenerate_identical_to_full"),
            "note": v.get("note"),
        })
    return rows


def query_level_rows(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix: ItemMatrix = study["matrix"]
    test_ids = study["test_ids"]
    oracle = study["oracle_test"]
    cost_of = study["cost_of"]
    rows = []
    for name, v in study["variants"].items():
        for qid in test_ids:
            t = int(v["assignment"][qid])
            rows.append({
                "variant": name,
                "query_id": qid,
                "benchmark": matrix.bench_of.get(qid, ""),
                "assigned_tier": t,
                "oracle_tier": int(oracle[qid]),
                "correct": int(bool(matrix.correct[(t, qid)])),
                "cost_usd": float(cost_of(t, qid)),
                "latency_s": float(matrix.latency_s[(t, qid)]),
                "regret_usd": float(cost_of(t, qid) - cost_of(oracle[qid], qid)),
            })
    return rows
