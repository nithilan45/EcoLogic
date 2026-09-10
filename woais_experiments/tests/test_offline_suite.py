import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.frozen import verify_frozen_hashes
from woais_experiments.paths import RESULTS
from woais_experiments.run_offline import run


class TestOfflineSuite(unittest.TestCase):
    def test_run_writes_results_and_preserves_hashes(self):
        summary = run()
        self.assertEqual(summary["n_items"], 364)
        self.assertTrue(summary["always_t2_dominates_ecologic_on_usd"])
        self.assertTrue(summary["regret_identity_ecologic_usd_reconciles"])
        self.assertTrue((RESULTS / "summary.json").exists())
        self.assertTrue((RESULTS / "accounting" / "stage12.json").exists())
        self.assertTrue((RESULTS / "routing" / "stage12_policies.json").exists())
        self.assertTrue((RESULTS / "latency" / "stage12_wallclock.json").exists())
        self.assertTrue((RESULTS / "latency" / "serverless_model.json").exists())
        self.assertTrue((RESULTS / "external" / "routellm_tables.json").exists())
        self.assertTrue((RESULTS / "workloads" / "frozen.json").exists())
        self.assertTrue((RESULTS / "figures" / "accuracy_vs_usd.png").exists())
        self.assertTrue((RESULTS / "figures" / "latency_cdf_by_tier.png").exists())
        self.assertTrue((RESULTS / "REPORT.md").exists())
        self.assertTrue((RESULTS / "accounting" / "framework" / "sign_flip_comparison.json").exists())
        self.assertTrue(verify_frozen_hashes()["ok"])
