import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.routing.policies import (
    accuracy_optimal_static_mixture,
    always_tier,
    cost_ordered_cascade,
    random_tiers,
)
from woais_experiments.statistics.inference import matched_cost_fraction, mixture_cost


class TestPolicies(unittest.TestCase):
    def test_always_tier_constant(self):
        items = ["a", "b", "c"]
        self.assertEqual(always_tier(items, 2), {"a": 2, "b": 2, "c": 2})

    def test_random_seed_matches_stage12(self):
        items = [str(i) for i in range(20)]
        a = random_tiers(items, seed=20260905)
        b = random_tiers(items, seed=20260905)
        c = random_tiers(items, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(set(a.values()) <= {1, 2, 3})

    def test_cascade_prefers_cheaper_when_both_clear_tau(self):
        p1 = np.array([0.9, 0.1, 0.9])
        p2 = np.array([0.9, 0.9, 0.1])
        t = cost_ordered_cascade(p1, p2, tau=0.5, order=(2, 1))
        self.assertEqual(list(t), [2, 2, 1])

    def test_cascade_falls_back_to_tier3(self):
        p1 = np.array([0.1])
        p2 = np.array([0.1])
        t = cost_ordered_cascade(p1, p2, tau=0.5, order=(2, 1))
        self.assertEqual(int(t[0]), 3)

    def test_matched_cost_fraction_endpoints(self):
        self.assertAlmostEqual(matched_cost_fraction(1.0, 1.0, 5.0), 0.0)
        self.assertAlmostEqual(matched_cost_fraction(5.0, 1.0, 5.0), 1.0)
        self.assertAlmostEqual(matched_cost_fraction(3.0, 1.0, 5.0), 0.5)
        self.assertAlmostEqual(mixture_cost(0.5, 1.0, 5.0), 3.0)

    def test_accuracy_optimal_mixture_picks_best_feasible_pure(self):
        # T2 is cheaper and more accurate than T1/T3: extra budget should not be spent.
        mean_cost = {1: 3.0, 2: 1.0, 3: 10.0}
        mean_acc = {1: 0.80, 2: 0.92, 3: 0.90}
        out = accuracy_optimal_static_mixture(mean_cost, mean_acc, budget=5.0)
        self.assertTrue(out["feasible"])
        self.assertAlmostEqual(out["mix"][2], 1.0)
        self.assertAlmostEqual(out["accuracy"], 0.92)

    def test_accuracy_optimal_mixture_mixes_when_needed(self):
        mean_cost = {1: 1.0, 2: 5.0}
        mean_acc = {1: 0.5, 2: 0.9}
        out = accuracy_optimal_static_mixture(mean_cost, mean_acc, budget=3.0)
        self.assertTrue(out["feasible"])
        self.assertAlmostEqual(out["cost"], 3.0, places=9)
        self.assertAlmostEqual(out["mix"][1], 0.5)
        self.assertAlmostEqual(out["mix"][2], 0.5)
        self.assertAlmostEqual(out["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
