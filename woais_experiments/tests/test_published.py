import json
import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.accounting.costs import cost_fn_usd
from woais_experiments.frozen import load_json, load_stage12_matrix, load_stage12_routing
from woais_experiments.paths import CONFIGS, ROOT as REPO
from woais_experiments.routing.policies import build_stage12_policies, evaluate_policies


class TestPublishedStage12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_stage12_matrix()
        cls.routing = load_stage12_routing()
        cls.policies = build_stage12_policies(cls.matrix, cls.routing)
        cls.usd = evaluate_policies(cls.matrix, cls.policies, cost_fn_usd(cls.matrix))
        cls.published = load_json(REPO / "raw_results" / "analysis_cost.json")
        cls.anchors = json.loads((CONFIGS / "experiment.json").read_text())["published_stage12"]

    def test_complete_item_count(self):
        self.assertEqual(self.matrix.n, 364)
        self.assertEqual(self.published["n_items"], 364)

    def test_policy_correct_counts(self):
        for name, key in (
            ("ecologic", "ecologic"),
            ("always_t1", "always_t1"),
            ("always_t2", "always_t2"),
            ("frontier", "frontier"),
            ("random", "random"),
            ("oracle_usd", "oracle"),
        ):
            self.assertEqual(
                self.usd["policies"][name]["correct"],
                self.published["policies"][key]["correct"],
                msg=name,
            )

    def test_policy_usd_totals(self):
        for name, key in (
            ("ecologic", "ecologic"),
            ("always_t1", "always_t1"),
            ("always_t2", "always_t2"),
            ("frontier", "frontier"),
            ("random", "random"),
            ("oracle_usd", "oracle"),
        ):
            self.assertAlmostEqual(
                self.usd["policies"][name]["cost"],
                self.published["policies"][key]["cost_usd"],
                places=8,
                msg=name,
            )

    def test_always_t2_dominates(self):
        eco = self.usd["policies"]["ecologic"]
        t2 = self.usd["policies"]["always_t2"]
        self.assertGreater(t2["accuracy"], eco["accuracy"])
        self.assertLess(t2["cost"], eco["cost"])
        ratio = eco["cost"] / t2["cost"]
        self.assertAlmostEqual(ratio, self.anchors["cost_ratio_ecologic_over_t2"], places=8)

    def test_mcnemar_vs_always_t2(self):
        mc = self.usd["mcnemar_vs"]["always_t2"]
        pub = self.published["mcnemar_vs_ecologic"]["always_t2"]
        self.assertEqual(mc["b"], pub["b"])
        self.assertEqual(mc["c"], pub["c"])
        self.assertAlmostEqual(mc["p_value"], pub["p_value"], places=10)


if __name__ == "__main__":
    unittest.main()
