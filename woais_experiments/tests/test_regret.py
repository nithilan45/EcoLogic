import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.statistics.regret import (
    decompose_regret,
    synthetic_length_biased_example,
    synthetic_query_independent_example,
)


class TestRegretIdentity(unittest.TestCase):
    def test_algebra_holds_on_random_assignments(self):
        rng = np.random.default_rng(0)
        n, m = 200, 3
        costs = rng.lognormal(size=(n, m))
        t_star = rng.integers(1, m + 1, size=n)
        t_hat = rng.integers(1, m + 1, size=n)
        out = decompose_regret(t_star, t_hat, costs, labels=[1, 2, 3])
        self.assertTrue(out["reconciles"])
        self.assertLess(out["abs_residual"], 1e-10)

    def test_zero_regret_when_hat_equals_star(self):
        rng = np.random.default_rng(1)
        n = 80
        costs = rng.random((n, 2))
        t = rng.integers(0, 2, size=n)
        out = decompose_regret(t, t.copy(), costs, labels=[0, 1])
        self.assertAlmostEqual(out["R_true"], 0.0, places=12)
        self.assertAlmostEqual(out["R_naive"], 0.0, places=12)
        self.assertAlmostEqual(out["correction"], 0.0, places=12)

    def test_length_bias_has_nonzero_correction(self):
        biased = synthetic_length_biased_example()
        independent = synthetic_query_independent_example()
        self.assertTrue(biased["reconciles"])
        self.assertTrue(independent["reconciles"])
        self.assertGreater(abs(biased["correction"]), abs(independent["correction"]))
        # Escalating long items makes naive look cheaper than truth.
        self.assertLess(biased["R_naive"], biased["R_true"])


if __name__ == "__main__":
    unittest.main()
