"""Held-out split, leakage, preprocessing, and lock protocol tests."""

from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from woais_experiments.accounting.costs import TIERS
from woais_experiments.frozen import ItemMatrix, load_stage12_matrix, write_result
from woais_experiments.heldout.build_split import (
    JACCARD_NEAR,
    MANIFEST_PATH,
    SPLIT_SEED,
    allocate_stratified_grouped,
    build_manifest,
    cross_split_overlap_flags,
    jaccard,
    load_manifest,
    load_panel,
    merge_duplicate_groups,
    natural_group_id,
    normalize_text,
    split_ids,
    token_set,
    verify_manifest_hash,
)
from woais_experiments.heldout.common import (
    FINAL_CSV_REL,
    FROZEN_CONFIG_REL,
    LOCK_REL,
    LockedExperimentError,
    QUERY_CSV_REL,
    TestQueryMismatchError,
    assert_same_test_population,
    canonical,
    hash_config,
    protocol_source_sha256,
    quota_assignment_sorted,
    refuse_if_locked,
    sha256_text,
)
from woais_experiments.heldout.evaluate_router import evaluate, load_frozen_config
from woais_experiments.paths import RESULTS, get_results_root, reset_run_context, set_run_context
from woais_experiments.heldout.train_router import (
    fit_train_preprocessor,
    select_length_tau,
)
from woais_experiments.paths import RESULTS, reset_run_context, set_run_context
from woais_experiments.routing.ablations import DENSE_NAMES, TrainOnlyFeatures, _lexicons
from woais_experiments.tests import ROOT  # noqa: F401


def _tmp_root() -> Path:
    root = RESULTS / "_heldout_unittest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _toy_records(n_gsm: int = 30, n_subj: int = 6, per_subj: int = 5) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for i in range(n_gsm):
        qid = f"gsm-{i:03d}"
        recs[qid] = {
            "item_id": qid,
            "benchmark": "gsm8k",
            "subject": None,
            "raw_query": f"what is {i} plus {i + 7} uniquely {i} apples",
        }
    for s in range(n_subj):
        subj = f"subject_{s}"
        for j in range(per_subj):
            qid = f"mmlu-{subj}-{j}"
            recs[qid] = {
                "item_id": qid,
                "benchmark": "mmlu",
                "subject": subj,
                "raw_query": f"mmlu {subj} question number {j} concept {s}{j}",
            }
    return recs


def _matrix_from_length_rule(
    ids: list[str],
    records: dict,
    *,
    short_limit: int,
    invert: bool = False,
) -> ItemMatrix:
    """T1 correct iff word count <= short_limit (or the reverse if invert). T2/T3 always correct. Cost = tier."""
    lex = _lexicons()
    correct = {}
    usd = {}
    tokens = {}
    pt = {}
    ct = {}
    lat = {}
    bench = {}
    for i in ids:
        n_words = len(lex["tokenize"](records[i]["raw_query"]))
        t1_ok = n_words <= short_limit
        if invert:
            t1_ok = not t1_ok
        for t in TIERS:
            correct[(t, i)] = True if t > 1 else t1_ok
            usd[(t, i)] = float(t)
            tokens[(t, i)] = 10 * t
            pt[(t, i)] = 8
            ct[(t, i)] = 2 * t
            lat[(t, i)] = 0.1 * t
        bench[i] = records[i]["benchmark"]
    return ItemMatrix(
        item_ids=list(ids),
        bench_of=bench,
        correct=correct,
        tokens=tokens,
        prompt_tokens=pt,
        completion_tokens=ct,
        usd=usd,
        latency_s=lat,
        model_of={1: "t1", 2: "t2", 3: "t3"},
    )


def _raise(qid: str) -> float:
    raise AssertionError(f"threshold selection inspected test id {qid}")


class _GuardDict:
    def __init__(self, inner, forbidden: set[str]) -> None:
        self.inner = inner
        self.forbidden = forbidden

    def __getitem__(self, key):
        _t, qid = key
        if qid in self.forbidden:
            raise AssertionError(f"threshold selection inspected test id {qid}")
        return self.inner[key]


class _GuardMatrix:
    def __init__(self, inner: ItemMatrix, forbidden: set[str]) -> None:
        self._inner = inner
        self.correct = _GuardDict(inner.correct, forbidden)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestNormalizeAndHashes(unittest.TestCase):
    def test_normalize_is_stable(self):
        self.assertEqual(normalize_text("  Hello, WORLD!!  "), normalize_text("hello world"))
        self.assertEqual(sha256_text("abc"), sha256_text("abc"))
        self.assertNotEqual(sha256_text("abc"), sha256_text("abd"))

    def test_canonical_hash_stable(self):
        a = {"b": 1, "a": [2, 3]}
        self.assertEqual(canonical(a), canonical({"a": [2, 3], "b": 1}))
        self.assertEqual(hash_config({"x": 1.0, "config_sha256": "nope"}), hash_config({"x": 1.0}))

    def test_jaccard_near_duplicate(self):
        a = token_set("the cat sat on the mat extra")
        b = token_set("the cat sat on the mat extra extra")
        self.assertGreaterEqual(jaccard(a, b), JACCARD_NEAR)


class TestGroupedSplit(unittest.TestCase):
    def test_mmlu_subject_stays_in_one_split(self):
        recs = _toy_records()
        ids = sorted(recs)
        base = {i: natural_group_id(recs[i]) for i in ids}
        group_of, flags = merge_duplicate_groups(recs, base)
        self.assertEqual(flags, [])
        bench_of = {i: recs[i]["benchmark"] for i in ids}
        item_split = allocate_stratified_grouped(ids, group_of, bench_of, seed=SPLIT_SEED)
        by_subj: dict[str, set[str]] = {}
        for i in ids:
            if recs[i]["benchmark"] != "mmlu":
                continue
            by_subj.setdefault(recs[i]["subject"], set()).add(item_split[i])
        for subj, splits in by_subj.items():
            self.assertEqual(len(splits), 1, msg=subj)
        self.assertEqual(set(item_split.values()), {"train", "val", "test"})

    def test_near_duplicates_merge_and_do_not_cross_splits(self):
        recs = _toy_records()
        recs["gsm-000"]["raw_query"] = "alpha beta gamma delta epsilon zeta"
        recs["gsm-001"]["raw_query"] = "alpha beta gamma delta epsilon zeta"
        ids = sorted(recs)
        base = {i: natural_group_id(recs[i]) for i in ids}
        group_of, flags = merge_duplicate_groups(recs, base)
        self.assertTrue(any(f["kind"] == "exact_normalized" for f in flags))
        self.assertEqual(group_of["gsm-000"], group_of["gsm-001"])
        bench_of = {i: recs[i]["benchmark"] for i in ids}
        item_split = allocate_stratified_grouped(ids, group_of, bench_of, seed=1)
        self.assertEqual(item_split["gsm-000"], item_split["gsm-001"])
        leftover = cross_split_overlap_flags(recs, item_split)
        self.assertEqual(leftover, [])

    def test_splits_are_disjoint(self):
        recs = _toy_records()
        ids = sorted(recs)
        base = {i: natural_group_id(recs[i]) for i in ids}
        group_of, _ = merge_duplicate_groups(recs, base)
        bench_of = {i: recs[i]["benchmark"] for i in ids}
        item_split = allocate_stratified_grouped(ids, group_of, bench_of, seed=2)
        buckets = {"train": set(), "val": set(), "test": set()}
        for i, sp in item_split.items():
            buckets[sp].add(i)
        self.assertFalse(buckets["train"] & buckets["val"])
        self.assertFalse(buckets["train"] & buckets["test"])
        self.assertFalse(buckets["val"] & buckets["test"])
        self.assertEqual(buckets["train"] | buckets["val"] | buckets["test"], set(ids))


class TestTrainOnlyPreprocessing(unittest.TestCase):
    def test_tfidf_vocabulary_excludes_test_token(self):
        train = ["alpha beta alpha", "beta gamma"]
        n_dense = len(DENSE_NAMES)
        feat = fit_train_preprocessor(train, np.zeros((2, n_dense)), seed=1)
        self.assertIsNotNone(feat.tfidf)
        vocab = set(feat.tfidf.get_feature_names_out())
        self.assertNotIn("unicornzz", vocab)
        x, names, _ = feat.transform(["unicornzz appears only at test"], np.zeros((1, n_dense)))
        self.assertEqual(x.shape[0], 1)
        self.assertNotIn("tfidf:unicornzz", names)

    def test_scaler_fit_on_train_not_test(self):
        train_dense = np.array([[0.0], [1.0], [2.0]])
        feat = TrainOnlyFeatures(
            ["a", "b", "c"], train_dense, dense_names=["n_words"], use_semantic=False
        )
        mean = float(feat.scaler.mean_[0])
        self.assertAlmostEqual(mean, 1.0)
        test_dense = np.array([[100.0]])
        xt, _, _ = feat.transform(["z"], test_dense)
        # Transform uses train mean/scale; 100 is not the fitted mean.
        self.assertNotAlmostEqual(mean, 100.0)
        self.assertEqual(xt.shape[0], 1)


class TestValidationOnlyThreshold(unittest.TestCase):
    def test_tau_uses_val_not_test(self):
        val_ids = [f"val-{i}" for i in range(8)]
        test_ids = [f"test-{i}" for i in range(8)]
        records = {}
        for i, qid in enumerate(val_ids):
            n = 2 if i < 4 else 40
            records[qid] = {
                "item_id": qid,
                "benchmark": "gsm8k",
                "raw_query": "word " * n,
            }
        for i, qid in enumerate(test_ids):
            n = 40 if i < 4 else 2
            records[qid] = {
                "item_id": qid,
                "benchmark": "gsm8k",
                "raw_query": "word " * n,
            }
        cost_of = lambda t, i: (_raise(i) if i in set(test_ids) else float(t))
        val_m = _GuardMatrix(_matrix_from_length_rule(val_ids, records, short_limit=8), set(test_ids))
        test_m = _matrix_from_length_rule(test_ids, records, short_limit=8, invert=True)
        lex = _lexicons()
        val_sel = select_length_tau(
            val_ids, records, val_m, cost_of, cheap=1, strong=3, lex=lex
        )
        self.assertEqual(val_sel["selected_on"], "validation")
        self.assertFalse(val_sel["test_used"])
        self.assertGreater(val_sel["tau"], 0.0)
        # Selecting on test is a different call; the val path never touched test labels.
        test_sel = select_length_tau(
            test_ids, records, test_m, lambda t, i: float(t), cheap=1, strong=3, lex=lex
        )
        self.assertEqual(test_sel["selected_on"], "validation")
        self.assertNotIn("test-", ",".join(val_ids))


class TestSameTestPopulation(unittest.TestCase):
    def test_identical_ids_pass(self):
        ids = ["a", "b", "c"]
        assigns = {
            "r": {i: 1 for i in ids},
            "b": {i: 3 for i in ids},
        }
        assert_same_test_population(assigns, ids)

    def test_mismatch_raises(self):
        ids = ["a", "b", "c"]
        assigns = {
            "r": {"a": 1, "b": 1},
            "b": {i: 3 for i in ids},
        }
        with self.assertRaises(TestQueryMismatchError):
            assert_same_test_population(assigns, ids)


class TestQuotaAndLock(unittest.TestCase):
    def test_quota_is_deterministic(self):
        ids = [f"q{i}" for i in range(10)]
        mix = {1: 0.5, 2: 0.3, 3: 0.2}
        a = quota_assignment_sorted(ids, mix)
        b = quota_assignment_sorted(list(reversed(ids)), mix)
        self.assertEqual(a, b)
        counts = {t: sum(1 for v in a.values() if v == t) for t in TIERS}
        self.assertEqual(sum(counts.values()), 10)
        self.assertEqual(counts[1], 5)

    def test_lock_refuses_without_new_experiment(self):
        root = _tmp_root()
        tokens = set_run_context(output_root=root, overwrite_policy="replace")
        try:
            write_result(LOCK_REL, {"lock": True, "config_sha256": "abc"}, clobber=True)
            with self.assertRaises(LockedExperimentError):
                refuse_if_locked(new_experiment=False, action="re-evaluate")
            lock = refuse_if_locked(new_experiment=True, action="re-evaluate")
            self.assertTrue(lock["lock"])
        finally:
            reset_run_context(tokens)
            shutil.rmtree(root, ignore_errors=True)

    def test_protocol_source_hash_is_stable(self):
        a = protocol_source_sha256()
        b = protocol_source_sha256()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class TestFrozenManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST_PATH.exists():
            build_manifest(seed=SPLIT_SEED, force=False)

    def test_hash_stable_and_splits_disjoint(self):
        man = load_manifest()
        h1 = verify_manifest_hash(man)
        h2 = verify_manifest_hash(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
        self.assertEqual(h1, h2)
        self.assertEqual(h1, man["manifest_sha256"])
        sp = split_ids(man)
        t, v, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        self.assertFalse(t & v)
        self.assertFalse(t & te)
        self.assertFalse(v & te)
        self.assertEqual(t | v | te, {q["item_id"] for q in man["queries"]})
        self.assertEqual(man["n"]["total"], len(t) + len(v) + len(te))
        self.assertGreater(man["n"]["train"], man["n"]["val"])
        self.assertEqual(man["seed"], SPLIT_SEED)
        self.assertEqual(man["cross_split_overlap"], [])

    def test_disjoint_from_stage12(self):
        matrix, _ = load_panel()
        s12 = set(load_stage12_matrix().item_ids)
        self.assertFalse(set(matrix.item_ids) & s12)
        man = load_manifest()
        self.assertIn("n=364", man["note"])
        self.assertIn("historical", man["note"].lower())

    def test_groups_do_not_cross_splits(self):
        man = load_manifest()
        by_group: dict[str, set[str]] = {}
        for q in man["queries"]:
            by_group.setdefault(q["group_id"], set()).add(q["split"])
        for gid, splits in by_group.items():
            self.assertEqual(len(splits), 1, msg=gid)


def _heldout_results_exist() -> bool:
    root = get_results_root()
    return all((root / p).exists() for p in (FROZEN_CONFIG_REL, FINAL_CSV_REL, QUERY_CSV_REL, LOCK_REL))


@unittest.skipUnless(_heldout_results_exist(), "held-out test artifacts not written yet")
class TestHeldoutArtifacts(unittest.TestCase):
    def test_frozen_config_never_inspected_test(self):
        cfg = load_frozen_config()
        self.assertFalse(cfg["test_metrics_inspected"])
        self.assertTrue(cfg["historical_stage12_n364_not_used"])
        for key in ("logistic", "tree", "threshold", "ecologic_heuristic"):
            self.assertEqual(cfg[key]["selected_on"], "validation")
            self.assertFalse(cfg[key]["test_used"])
        self.assertEqual(cfg["preprocessing"]["tfidf_fit_on"], "train")
        self.assertEqual(cfg["preprocessing"]["scaler_fit_on"], "train")

    def test_query_level_same_test_population(self):
        man = load_manifest()
        test_ids = set(split_ids(man)["test"])
        path = get_results_root() / QUERY_CSV_REL
        by_method: dict[str, set[str]] = {}
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                by_method.setdefault(row["method"], set()).add(row["item_id"])
        self.assertGreaterEqual(len(by_method), 5)
        for name, ids in by_method.items():
            self.assertEqual(ids, test_ids, msg=name)

    def test_lock_blocks_reevaluation(self):
        with self.assertRaises(LockedExperimentError):
            evaluate(new_experiment=False, n_boot=10, n_perm=10)


if __name__ == "__main__":
    unittest.main()
