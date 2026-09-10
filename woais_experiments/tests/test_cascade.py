import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.paths import ensure_legacy_imports
from woais_experiments.routing.policies import cost_ordered_cascade


class TestCascadeMatchesLegacy(unittest.TestCase):
    def test_local_route_matches_calibrate_route(self):
        ensure_legacy_imports()
        try:
            from calibrate import route
        except Exception as exc:  # pragma: no cover - training stack optional
            self.skipTest(f"calibrate.route unavailable: {exc}")
        rng = np.random.default_rng(0)
        p1 = rng.random(64)
        p2 = rng.random(64)
        for tau in (0.1, 0.5, 0.9):
            for order in ((2, 1), (1, 2)):
                a = cost_ordered_cascade(p1, p2, tau, order)
                b = route(p1, p2, tau, order)
                np.testing.assert_array_equal(a, b)
