"""Paired bootstrap, permutation, Wilcoxon, effect sizes, and BH FDR."""

from __future__ import annotations

import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.statistics.bootstrap import (
    DEFAULT_N_BOOT,
    SIGNIFICANCE_NOTE,
    paired_bootstrap_effects,
    paired_bootstrap_mean,
)
from woais_experiments.statistics.effect_sizes import (
    cliffs_delta,
    cohens_dz,
    finite_pairs,
    mean_paired_difference,
    median_paired_difference,
    pair_by_query_id,
)
from woais_experiments.statistics.paired_tests import (
    apply_bh,
    benjamini_hochberg,
    compare_many,
    paired_comparison,
    paired_permutation_test,
    wilcoxon_signed_rank,
)


REQUIRED = ("n", "estimate", "ci_lo", "ci_hi", "p_raw", "p_adjusted", "effect_size")


def _shifted_pairs(n=40, shift=1.5, seed=20260909):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n)
    a = b + shift + 0.05 * rng.normal(size=n)
    return a, b


class TestEffectSizes(unittest.TestCase):
    def test_mean_median_and_drops_nan(self):
        a = [1.0, 2.0, np.nan, 4.0]
        b = [0.0, 2.0, 1.0, 1.0]
        aa, bb, meta = finite_pairs(a, b)
        self.assertEqual(meta["n"], 3)
        self.assertEqual(meta["n_dropped"], 1)
        self.assertAlmostEqual(mean_paired_difference(a, b), (1 + 0 + 3) / 3)
        self.assertAlmostEqual(median_paired_difference([1, 2, 3], [0, 0, 0]), 2.0)

    def test_pair_by_query_id_inner_join(self):
        a = {"q1": 1.0, "q2": 2.0, "q3": np.nan}
        b = {"q1": 0.0, "q2": 2.0, "q4": 9.0}
        aa, bb, ids, meta = pair_by_query_id(a, b)
        self.assertEqual(ids, ("q1", "q2"))
        self.assertEqual(meta["n_unmatched_a"], 1)
        self.assertEqual(meta["n_unmatched_b"], 1)
        self.assertEqual(meta["n"], 2)

    def test_cohens_dz_known(self):
        a = np.array([3.0, 3.0, 5.0, 5.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])
        d = a - b
        got = cohens_dz(a, b)
        self.assertTrue(got["available"])
        self.assertAlmostEqual(got["estimate"], float(d.mean() / d.std(ddof=1)))

    def test_cohens_dz_undefined_zero_variance(self):
        a = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        got = cohens_dz(a, b)
        self.assertFalse(got["available"])
        self.assertIsNone(got["estimate"])

    def test_cliffs_delta_paired_extremes(self):
        a = np.array([3.0, 4.0, 5.0, 5.0])
        b = np.array([1.0, 1.0, 1.0, 5.0])
        d = cliffs_delta(a, b, paired=True)
        self.assertEqual(d["n_positive"], 3)
        self.assertEqual(d["n_ties"], 1)
        self.assertAlmostEqual(d["estimate"], 3 / 4)
        all_less = cliffs_delta([0, 0, 0], [1, 2, 3], paired=True)
        self.assertAlmostEqual(all_less["estimate"], -1.0)

    def test_unpaired_cliffs_not_used_by_default(self):
        a = [1.0, 100.0]
        b = [0.0, 50.0]
        paired = cliffs_delta(a, b, paired=True)
        unpaired = cliffs_delta(a, b, paired=False)
        self.assertEqual(paired["n"], 2)
        self.assertEqual(unpaired["n"], 4)
        self.assertAlmostEqual(paired["estimate"], 1.0)
        self.assertAlmostEqual(unpaired["estimate"], 0.5)


class TestBootstrapAndPermutation(unittest.TestCase):
    def test_default_n_boot_is_10000(self):
        self.assertEqual(DEFAULT_N_BOOT, 10_000)

    def test_identical_series_mean_zero_high_p(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = paired_comparison(x, x, n_boot=200, n_perm=200, seed=1)
        self.assertAlmostEqual(out["mean_paired_difference"]["estimate"], 0.0)
        self.assertGreater(out["mean_paired_difference"]["p_raw"], 0.2)
        self.assertIn(SIGNIFICANCE_NOTE, out["significance_note"])
        for key in ("mean_paired_difference", "median_paired_difference", "cohens_dz", "cliffs_delta"):
            row = out[key]
            for field in REQUIRED:
                self.assertIn(field, row)
            self.assertIsNone(row["p_adjusted"])

    def test_constant_shift_ci_covers_and_p_small(self):
        a, b = _shifted_pairs()
        out = paired_comparison(a, b, n_boot=400, n_perm=400, seed=20260909)
        mean = out["mean_paired_difference"]
        self.assertAlmostEqual(mean["estimate"], float((a - b).mean()), places=12)
        self.assertLessEqual(mean["ci_lo"], mean["estimate"])
        self.assertGreaterEqual(mean["ci_hi"], mean["estimate"])
        self.assertTrue(0.0 < mean["p_raw"] <= 1.0)
        self.assertLess(mean["p_raw"], 0.05)
        self.assertGreater(out["cohens_dz"]["estimate"], 1.0)
        self.assertGreater(out["cliffs_delta"]["estimate"], 0.5)

    def test_bootstrap_reproducible(self):
        a, b = _shifted_pairs(n=20)
        x = paired_bootstrap_mean(a, b, n_boot=150, seed=7)
        y = paired_bootstrap_mean(a, b, n_boot=150, seed=7)
        self.assertEqual(x["ci_lo"], y["ci_lo"])
        self.assertEqual(x["ci_hi"], y["ci_hi"])

    def test_exact_permutation_on_tiny_n(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        perm = paired_permutation_test(a, b, "mean", n_perm=10, seed=0)
        self.assertEqual(perm["method"], "exact_sign_flip")
        self.assertEqual(perm["n_perm"], 8)
        # All-positive differences: two-sided exact p = 2 / 8 = 0.25
        self.assertAlmostEqual(perm["p_raw"], 0.25)

    def test_wilcoxon_matches_scipy_and_skips_all_zero(self):
        from scipy.stats import wilcoxon

        a, b = _shifted_pairs(n=30, shift=0.8)
        d = a - b
        ours = wilcoxon_signed_rank(a, b)
        try:
            ref = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
        except TypeError:
            ref = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        self.assertTrue(ours["available"])
        self.assertAlmostEqual(ours["p_raw"], float(ref.pvalue), places=10)
        z = wilcoxon_signed_rank([1, 1, 1], [1, 1, 1])
        self.assertFalse(z["available"])
        self.assertIn("zero", z["reason"])

    def test_missing_not_imputed(self):
        a = [1.0, np.nan, 3.0]
        b = [0.0, 5.0, 1.0]
        out = paired_comparison(a, b, n_boot=50, n_perm=50, seed=0)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_dropped"], 1)


class TestBHAndSchema(unittest.TestCase):
    def test_bh_known_values(self):
        # sorted 0.005, 0.01, 0.03, 0.04 → adj 0.02, 0.02, 0.04, 0.04
        p = [0.01, 0.04, 0.03, 0.005]
        bh = benjamini_hochberg(p)
        adj = bh["p_adjusted"]
        self.assertAlmostEqual(adj[3], 0.02)
        self.assertAlmostEqual(adj[0], 0.02)
        self.assertAlmostEqual(adj[2], 0.04)
        self.assertAlmostEqual(adj[1], 0.04)
        try:
            from scipy.stats import false_discovery_control

            ref = false_discovery_control(np.array(p), method="bh")
            for a, r in zip(adj, ref):
                self.assertAlmostEqual(a, float(r), places=12)
        except Exception:
            pass

    def test_bh_skips_nonfinite(self):
        bh = benjamini_hochberg([0.01, None, float("nan"), 0.04])
        self.assertEqual(bh["n_valid"], 2)
        self.assertIsNone(bh["p_adjusted"][1])

    def test_compare_many_fills_adjusted(self):
        rng = np.random.default_rng(0)
        comps = []
        for i, shift in enumerate((0.0, 2.0, 0.1)):
            b = rng.normal(size=24)
            a = b + shift
            comps.append({"name": f"s{i}", "a": a, "b": b, "name_a": "a", "name_b": "b"})
        out = compare_many(comps, n_boot=80, n_perm=80, seed=1, alpha=0.05)
        self.assertEqual(out["n_comparisons"], 3)
        ps = [c["mean_paired_difference"]["p_adjusted"] for c in out["comparisons"]]
        self.assertTrue(all(p is not None for p in ps))
        raw = [c["mean_paired_difference"]["p_raw"] for c in out["comparisons"]]
        # The large shift should have the smallest p, still smallest after BH.
        self.assertEqual(int(np.argmin(ps)), int(np.argmin(raw)))

    def test_apply_bh_does_not_use_ci_overlap(self):
        rows = [
            {"p_raw": 0.04, "ci_excludes_zero": False, "n": 10, "estimate": 0.1, "ci_lo": -0.01, "ci_hi": 0.2, "effect_size": 0.2, "p_adjusted": None},
            {"p_raw": 0.001, "ci_excludes_zero": True, "n": 10, "estimate": 1.0, "ci_lo": 0.5, "ci_hi": 1.5, "effect_size": 1.0, "p_adjusted": None},
        ]
        adj = apply_bh(rows)
        self.assertIsNotNone(adj[0]["p_adjusted"])
        self.assertNotIn("significant", adj[0])


class TestStage12Paired(unittest.TestCase):
    def test_always_t2_vs_ecologic_quality(self):
        from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
        from woais_experiments.routing.policies import keyword_from_routing

        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        eco = keyword_from_routing(routing, matrix.item_ids, "raw")
        acc_t2 = np.array([float(matrix.correct[(2, i)]) for i in matrix.item_ids])
        acc_eco = np.array([float(matrix.correct[(eco[i], i)]) for i in matrix.item_ids])
        out = paired_comparison(
            acc_t2,
            acc_eco,
            query_ids=matrix.item_ids,
            name_a="always_t2",
            name_b="ecologic",
            n_boot=200,
            n_perm=200,
            seed=20260909,
        )
        self.assertEqual(out["n"], 364)
        self.assertGreater(out["mean_paired_difference"]["estimate"], 0.0)
        self.assertLess(out["mean_paired_difference"]["p_raw"], 0.05)
        self.assertIsNone(out["mean_paired_difference"]["p_adjusted"])
        self.assertIn("ci_excludes_zero", out["mean_paired_difference"])
        self.assertIn("Do not reject", out["significance_note"])


if __name__ == "__main__":
    unittest.main()
