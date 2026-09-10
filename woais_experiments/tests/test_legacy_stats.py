import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.paths import ensure_legacy_imports
from woais_experiments.statistics.inference import mcnemar, wilson


class TestLegacyStatsMatch(unittest.TestCase):
    def test_wilson_and_mcnemar_match_analyze_when_importable(self):
        ensure_legacy_imports()
        try:
            from analyze import mcnemar as legacy_mcnemar
            from analyze import wilson as legacy_wilson
        except Exception as exc:
            self.skipTest(f"benchmark.analyze not importable: {exc}")
        for k, n in ((316, 364), (0, 10), (10, 10), (1, 2)):
            self.assertEqual(wilson(k, n), legacy_wilson(k, n))
        a = [True, True, False, False, True]
        b = [True, False, True, False, True]
        self.assertEqual(mcnemar(a, b), legacy_mcnemar(a, b))
