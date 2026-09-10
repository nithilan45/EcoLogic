import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.external.routellm import summarize_s9
from woais_experiments.frozen import load_json
from woais_experiments.paths import CONFIGS, ROOT as REPO
import json


class TestExternalRouteLLM(unittest.TestCase):
    def test_s9_summary_matches_committed_json(self):
        s9 = load_json(REPO / "stage7_10" / "s9_static_baselines.json")
        summary = summarize_s9(s9)
        anchors = json.loads((CONFIGS / "experiment.json").read_text())["published_stage9"]
        self.assertEqual(summary["n_items"], anchors["n_items"])
        self.assertEqual(summary["n_interior"], anchors["n_interior"])
        self.assertEqual(
            summary["n_interior_beating_matched_cost"],
            anchors["n_interior_beating_matched_cost"],
        )
        self.assertEqual(summary["n_interior_significant_p05"], anchors["n_interior_significant"])
        self.assertAlmostEqual(
            summary["mean_edge_matched_cost_pp"],
            anchors["mean_router_edge_matched_cost_pp"],
            places=10,
        )
        self.assertAlmostEqual(summary["sign_test_p"], anchors["sign_test_p"], places=10)

    def test_does_not_require_routellm_clone(self):
        summary = summarize_s9()
        self.assertIn("/tmp/routellm_chk is not required", summary["note"])


if __name__ == "__main__":
    unittest.main()
