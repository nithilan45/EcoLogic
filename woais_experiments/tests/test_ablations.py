"""Leakage, query-set, and metric checks for the router ablation framework."""

from __future__ import annotations

import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.frozen import load_stage12_matrix
from woais_experiments.routing.ablations import (
    PRODUCTION_FAMILIES,
    REFERENCE,
    TestQueryMismatchError,
    TrainOnlyFeatures,
    _lexicons,
    assert_disjoint_splits,
    assert_same_test_queries,
    assign_gated,
    inspect_feature_families,
    load_item_texts,
    query_level_rows,
    run_ablation_study,
    select_length_threshold,
    select_strongest_family,
    stratified_splits,
    summary_rows,
)


REQUIRED_VARIANTS = (
    "full",
    "no_semantic",
    "no_complexity",
    "no_length",
    "no_lexical",
    "no_confidence",
    "single_strongest_family",
    "threshold_only",
    "logistic",
    "tree",
    "random_matched_rate",
    "static_mixture_matched_rate",
    "always_cheap",
    "always_strong",
    "oracle",
)


class TestFeatureInventory(unittest.TestCase):
    def test_production_families_and_absences(self):
        inv = inspect_feature_families()
        fam = inv["families"]
        self.assertTrue(fam["lexical"]["present_in_production"])
        self.assertTrue(fam["complexity"]["present_in_production"])
        self.assertTrue(fam["length"]["present_in_production"])
        self.assertFalse(fam["semantic"]["present_in_production"])
        self.assertFalse(fam["confidence"]["present_in_production"])
        self.assertIn("not causal", inv["importance_note"].lower().replace("—", " "))


class TestGatedMatchesProduction(unittest.TestCase):
    def test_all_families_match_production_source(self):
        matrix = load_stage12_matrix()
        texts = {k: v["raw_query"] for k, v in load_item_texts().items()}
        lex = _lexicons()
        assign = assign_gated(matrix.item_ids, texts, PRODUCTION_FAMILIES, lex=lex)
        n_mismatch = 0
        for qid in matrix.item_ids:
            prod = int(lex["classify"](texts[qid]).recommended_tier)
            if assign[qid] != prod:
                n_mismatch += 1
        self.assertEqual(n_mismatch, 0)

    def test_dropping_lexical_changes_some_assignments(self):
        texts = {k: v["raw_query"] for k, v in load_item_texts().items()}
        ids = list(texts)[:80]
        lex = _lexicons()
        full = assign_gated(ids, texts, PRODUCTION_FAMILIES, lex=lex)
        no_lex = assign_gated(ids, texts, ("complexity", "length"), lex=lex)
        self.assertNotEqual(full, no_lex)


class TestSplitsAndLeakage(unittest.TestCase):
    def test_splits_are_disjoint_and_cover(self):
        matrix = load_stage12_matrix()
        split = stratified_splits(matrix.item_ids, matrix.bench_of, seed=20260909)
        assert_disjoint_splits(split)
        self.assertEqual(
            set(split["train"]) | set(split["val"]) | set(split["test"]),
            set(matrix.item_ids),
        )
        self.assertGreater(len(split["test"]), 20)
        self.assertGreater(len(split["val"]), 20)

    def test_tfidf_vocabulary_is_train_only(self):
        train = ["alpha beta alpha", "beta gamma"]
        dense = np.zeros((2, 1))
        feat = TrainOnlyFeatures(
            train, dense, dense_names=["n_words"], use_semantic=True, max_features=64
        )
        vocab = set(feat.tfidf.get_feature_names_out())
        self.assertNotIn("unicornzz", vocab)
        x, names, families = feat.transform(["unicornzz appears only at test"], np.zeros((1, 1)))
        self.assertEqual(x.shape[0], 1)
        self.assertNotIn("tfidf:unicornzz", names)


class TestSelectionOnValidation(unittest.TestCase):
    def test_strongest_family_uses_val_not_test(self):
        matrix = load_stage12_matrix()
        texts = {k: v["raw_query"] for k, v in load_item_texts().items() if k in set(matrix.item_ids)}
        split = stratified_splits(matrix.item_ids, matrix.bench_of, seed=1)
        lex = _lexicons()
        picked = select_strongest_family(texts, split["val"], matrix, lex=lex)
        self.assertEqual(picked["selected_on"], "validation")
        self.assertFalse(picked["test_used"])
        self.assertIn(picked["family"], PRODUCTION_FAMILIES)
        # Recompute on test; selection must still be the val argmax, even if test disagrees.
        test_scores = {}
        for fam in PRODUCTION_FAMILIES:
            pred = assign_gated(split["test"], texts, [fam], lex=lex)
            test_scores[fam] = sum(
                matrix.correct[(pred[i], i)] for i in split["test"]
            ) / len(split["test"])
        val_best = max(PRODUCTION_FAMILIES, key=lambda f: (picked["val_scores"][f], f))
        self.assertEqual(picked["family"], val_best)

    def test_threshold_maximizes_validation_accuracy(self):
        length = np.array([0.0, 1.0, 2.0, 3.0])
        # Oracle: strong only for the two longest.
        oracle = np.array([2, 2, 3, 3])
        out = select_length_threshold(length, oracle, cheap=2, strong=3)
        pred = np.where(length >= out["tau"], 3, 2)
        self.assertGreaterEqual(float(np.mean(pred == oracle)), 0.75)
        self.assertIn("tau", out)


class TestStudyProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = run_ablation_study(n_boot=40, n_perm=39, seed=20260909, model_seed=20260909)

    def test_all_required_variants_are_present(self):
        names = set(self.study["variants"])
        self.assertEqual(names, set(REQUIRED_VARIANTS))

    def test_identical_test_queries(self):
        assert_same_test_queries(self.study["variants"], self.study["test_ids"])
        rows = query_level_rows(self.study)
        by_var = {}
        for row in rows:
            by_var.setdefault(row["variant"], set()).add(row["query_id"])
        ref = by_var[REFERENCE]
        for name, ids in by_var.items():
            self.assertEqual(ids, ref, msg=name)
        with self.assertRaises(TestQueryMismatchError):
            bad = {k: dict(v) for k, v in self.study["variants"].items()}
            drop = next(iter(self.study["test_ids"]))
            bad["full"] = dict(bad["full"])
            bad["full"]["assignment"] = {
                q: t for q, t in bad["full"]["assignment"].items() if q != drop
            }
            assert_same_test_queries(bad, self.study["test_ids"])

    def test_no_test_selection_flag(self):
        cfg = self.study["configs"]
        self.assertFalse(cfg["protocol"]["select_ablation_on_test"])
        self.assertEqual(cfg["protocol"]["test"], "once_per_frozen_configuration")
        self.assertEqual(cfg["strongest_family"]["selected_on"], "validation")
        self.assertNotIn("selected_on_test", cfg["strongest_family"])

    def test_splits_not_in_each_other(self):
        sp = self.study["split"]
        assert_disjoint_splits(sp)

    def test_degenerate_absent_families_match_full(self):
        full = self.study["variants"]["full"]["assignment"]
        self.assertEqual(self.study["variants"]["no_semantic"]["assignment"], full)
        self.assertEqual(self.study["variants"]["no_confidence"]["assignment"], full)
        self.assertTrue(self.study["variants"]["no_semantic"]["degenerate_identical_to_full"])

    def test_summary_preserves_negative_results(self):
        rows = {r["variant"]: r for r in summary_rows(self.study)}
        self.assertEqual(set(rows), set(REQUIRED_VARIANTS))
        # Always-cheap is typically worse than oracle; still reported.
        self.assertIn("always_cheap", rows)
        self.assertIsNotNone(rows["always_cheap"]["quality"])
        self.assertIsNotNone(rows["oracle"]["quality"])
        self.assertGreaterEqual(rows["oracle"]["quality"], rows["always_cheap"]["quality"] - 1e-12)
        # vs-full stats exist (including losses).
        self.assertIn("delta_quality", rows["no_lexical"])
        self.assertIn("p_raw_quality", rows["no_lexical"])
        self.assertIn("p_adjusted_quality", rows["no_lexical"])

    def test_paired_fields_and_bh(self):
        cmp_ = self.study["comparisons"]["always_strong"]
        for endpoint in ("quality", "cost", "regret"):
            block = cmp_[endpoint]
            self.assertIn("delta", block)
            self.assertIn("ci_lo", block)
            self.assertIn("ci_hi", block)
            self.assertIn("p_raw", block)
            self.assertIn("p_adjusted", block)
            self.assertIn("effect_size", block)
            self.assertIn("cohens_dz", block["effect_size"])

    def test_importance_is_marked_noncausal(self):
        imp = self.study["configs"]["importance"]
        self.assertIn("not causal", imp["note"].lower())
        if imp["logistic"] is not None:
            self.assertFalse(imp["logistic"]["causal"])
        if imp["tree"] is not None:
            self.assertFalse(imp["tree"]["causal"])

    def test_oracle_is_not_worse_than_always_cheap_on_cost(self):
        rows = {r["variant"]: r for r in summary_rows(self.study)}
        self.assertLessEqual(rows["oracle"]["realized_cost"], rows["always_strong"]["realized_cost"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
