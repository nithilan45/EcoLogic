"""Break-even router overhead vs cost-matched static policies."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.accounting.breakeven import (
    analytic_usd_threshold,
    analyze_assignment,
    analyze_item_matrix,
    classify_status,
    classify_status_ci,
    numeric_min_cost_for_quality,
    numeric_usd_threshold,
    percentage_of_break_even_used,
    priced_token_break_even,
    rows_to_csv,
    table_from_policies,
    verify_usd_threshold,
)
from woais_experiments.accounting.per_query_cost import load_router_overhead
from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
from woais_experiments.routing.frontier import StaticMarket, interpolate_at_quality
from woais_experiments.routing.policies import build_stage12_policies


class TestClassifyAndPercentage(unittest.TestCase):
    def test_sign_of_net_savings(self):
        self.assertEqual(classify_status(0.2), "beneficial")
        self.assertEqual(classify_status(0.0), "neutral")
        self.assertEqual(classify_status(-1e-9), "dominated")
        self.assertEqual(classify_status(math.inf), "beneficial")
        self.assertEqual(classify_status(float("-inf")), "dominated")

    def test_ci_covers_zero_is_neutral(self):
        self.assertEqual(classify_status_ci(-0.1, 0.2), "neutral")
        self.assertEqual(classify_status_ci(0.01, 0.04), "beneficial")
        self.assertEqual(classify_status_ci(-0.4, -0.1), "dominated")

    def test_percentage_used(self):
        self.assertAlmostEqual(percentage_of_break_even_used(0.05, 0.20), 25.0)
        self.assertEqual(percentage_of_break_even_used(0.0, math.inf), 0.0)
        self.assertIsNone(percentage_of_break_even_used(0.0, -0.1))
        self.assertEqual(percentage_of_break_even_used(0.0, 0.0), 0.0)
        self.assertTrue(math.isinf(percentage_of_break_even_used(0.01, 0.0)))


class TestTwoModelAnalytic(unittest.TestCase):
    """A=(C=1, q=0.5), B=(C=3, q=0.9). Mix at q=0.7 costs 2."""

    def setUp(self):
        self.m = StaticMarket(["A", "B"], [1.0, 3.0], [0.5, 0.9])

    def test_quality_0_7_at_cost_1_8(self):
        row = analytic_usd_threshold(self.m, 1.8, 0.7)
        self.assertTrue(row["same_quality_feasible"])
        self.assertAlmostEqual(row["raw_router_savings"], 0.2)
        self.assertAlmostEqual(row["break_even_overhead"], 0.2)
        numeric = numeric_usd_threshold(self.m, 1.8, 0.7)
        self.assertAlmostEqual(numeric, 0.2, places=10)
        chk = verify_usd_threshold(self.m, 1.8, 0.7, 0.2)
        self.assertTrue(chk["verified"])
        self.assertLess(chk["abs_err"], 1e-10)

    def test_numeric_inverts_envelope(self):
        c_star = numeric_min_cost_for_quality(self.m, 0.7)
        self.assertAlmostEqual(c_star, interpolate_at_quality(self.m, 0.7).cost, places=10)

    def test_already_dominated(self):
        row = analytic_usd_threshold(self.m, 2.5, 0.7)
        self.assertAlmostEqual(row["raw_router_savings"], -0.5)

    def test_quality_above_hull_is_infinite(self):
        row = analytic_usd_threshold(self.m, 2.0, 0.95)
        self.assertFalse(row["same_quality_feasible"])
        self.assertTrue(math.isinf(row["break_even_overhead"]))
        self.assertGreater(row["break_even_overhead"], 0)
        self.assertTrue(math.isinf(numeric_usd_threshold(self.m, 2.0, 0.95)))

    def test_zero_gap_on_the_hull_mix(self):
        row = analytic_usd_threshold(self.m, 2.0, 0.7)
        self.assertAlmostEqual(row["raw_router_savings"], 0.0)


class TestThreeModelChord(unittest.TestCase):
    def test_quality_0_85_at_cost_2_5(self):
        m = StaticMarket(["A", "B", "C"], [1.0, 2.0, 4.0], [0.4, 0.8, 0.9])
        row = analytic_usd_threshold(m, 2.5, 0.85)
        self.assertAlmostEqual(row["break_even_overhead"], 0.5)
        chk = verify_usd_threshold(m, 2.5, 0.85, 0.5)
        self.assertTrue(chk["verified"])


class TestPanelAssignment(unittest.TestCase):
    def test_identical_mix_is_neutral(self):
        # Homogeneous panel + the 50/50 assignment is exactly the static mix.
        cost = np.tile([[1.0, 3.0]], (10, 1))
        quality = np.tile([[0.5, 0.9]], (10, 1))
        choice = np.array([0, 1] * 5)
        out = analyze_assignment(
            ["A", "B"], cost, quality, choice,
            latency=np.tile([[10.0, 30.0]], (10, 1)),
            tokens=np.tile([[100.0, 400.0]], (10, 1)),
            n_boot=0, verify=True,
        )
        self.assertAlmostEqual(out["usd"]["raw_router_savings"], 0.0, places=12)
        self.assertEqual(out["usd"]["status"], "neutral")
        self.assertTrue(out["usd"]["routing_is_neutral"])
        self.assertAlmostEqual(out["latency_ms"]["raw_router_savings"], 0.0, places=12)
        self.assertAlmostEqual(out["tokens"]["raw_router_savings"], 0.0, places=12)
        self.assertTrue(out["usd"]["numeric_verified"])
        self.assertTrue(out["latency_ms"]["numeric_verified"])

    def test_router_cheaper_at_same_quality(self):
        # Two queries: A beats mean-cost matching.
        # Q1: A=(1,1), B=(3,1). Q2: A=(1,0), B=(3,1).
        # Means A=(1, 0.5), B=(3, 1.0). Oracle [A,B]: C=2, Q=1.0.
        # Static at Q=1 is always B, cost 3. H*_usd = 1.
        cost = np.array([[1.0, 3.0], [1.0, 3.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        latency = np.array([[5.0, 40.0], [15.0, 40.0]])
        tokens = np.array([[10.0, 80.0], [30.0, 80.0]])
        out = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1]),
            latency=latency, tokens=tokens,
            overhead_usd=0.0, overhead_latency_ms=0.0, overhead_tokens=0.0,
            n_boot=200, seed=20260909, verify=True,
        )
        self.assertAlmostEqual(out["router_quality"], 1.0)
        self.assertAlmostEqual(out["router_realized_cost"], 2.0)
        self.assertAlmostEqual(out["usd"]["raw_router_savings"], 1.0)
        self.assertEqual(out["usd"]["status"], "beneficial")
        self.assertAlmostEqual(out["usd"]["percentage_of_break_even_used"], 0.0)
        self.assertTrue(out["usd"]["numeric_verified"])
        # Quality-matched static is always B: L=40, T=80. Router (5+40)/2=22.5, tokens (10+80)/2=45.
        self.assertAlmostEqual(out["latency_ms"]["static_mean"], 40.0)
        self.assertAlmostEqual(out["latency_ms"]["router_mean"], 22.5)
        self.assertAlmostEqual(out["latency_ms"]["break_even_overhead"], 17.5)
        self.assertEqual(out["latency_ms"]["status"], "beneficial")
        self.assertAlmostEqual(out["tokens"]["break_even_overhead"], 35.0)

    def test_overhead_eats_the_gap(self):
        cost = np.array([[1.0, 3.0], [1.0, 3.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        beneficial = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1]),
            overhead_usd=0.4, n_boot=0, verify=False, latency=None, tokens=None,
        )
        self.assertEqual(beneficial["usd"]["status"], "beneficial")
        self.assertAlmostEqual(beneficial["usd"]["net_savings"], 0.6)
        self.assertAlmostEqual(beneficial["usd"]["percentage_of_break_even_used"], 40.0)

        knife = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1]),
            overhead_usd=1.0, n_boot=0, verify=False,
        )
        self.assertEqual(knife["usd"]["status"], "neutral")
        self.assertAlmostEqual(knife["usd"]["net_savings"], 0.0)

        dead = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1]),
            overhead_usd=1.2, n_boot=0, verify=False,
        )
        self.assertEqual(dead["usd"]["status"], "dominated")
        self.assertAlmostEqual(dead["usd"]["net_savings"], -0.2)

    def test_priced_tokens_convert_usd_gap(self):
        self.assertAlmostEqual(priced_token_break_even(0.2, 1.0), 200_000.0)
        self.assertTrue(math.isinf(priced_token_break_even(0.2, 0.0)))
        cost = np.array([[1.0, 3.0], [1.0, 3.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        tokens = np.array([[10.0, 80.0], [30.0, 80.0]])
        out = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1]),
            tokens=tokens, token_usd_per_million=2.0, n_boot=0, verify=False,
        )
        self.assertAlmostEqual(out["tokens"]["priced_token_break_even"], 500_000.0)

    def test_bootstrap_reproducible_and_covers_point(self):
        cost = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        a = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1, 0, 1]),
            n_boot=400, seed=20260909, verify=False,
        )
        b = analyze_assignment(
            ["A", "B"], cost, quality, np.array([0, 1, 0, 1]),
            n_boot=400, seed=20260909, verify=False,
        )
        self.assertEqual(a["usd"]["break_even_lo"], b["usd"]["break_even_lo"])
        self.assertEqual(a["usd"]["break_even_hi"], b["usd"]["break_even_hi"])
        lo, hi, pt = a["usd"]["break_even_lo"], a["usd"]["break_even_hi"], a["usd"]["break_even_overhead"]
        self.assertLessEqual(lo, pt + 1e-12)
        self.assertGreaterEqual(hi, pt - 1e-12)
        self.assertIn(a["usd"]["status_ci"], {"beneficial", "neutral", "dominated"})

    def test_csv_headers(self):
        cost = np.array([[1.0, 3.0], [1.0, 3.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        out = analyze_assignment(["A", "B"], cost, quality, np.array([0, 1]), n_boot=0)
        text = rows_to_csv(out["rows"])
        self.assertIn("raw_router_savings", text.splitlines()[0])
        self.assertIn("percentage_of_break_even_used", text)
        self.assertIn("break_even_overhead", text)


class TestStage12Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        cls.matrix = matrix
        cls.policies = build_stage12_policies(matrix, routing, seed=20260905)

    def test_keyword_overhead_is_zero(self):
        oh = load_router_overhead("ecologic_keyword")
        self.assertEqual(oh.cost_usd, 0.0)
        self.assertEqual(oh.tokens, 0.0)

    def test_ecologic_dominated_always_t2_neutral_oracle_above_hull(self):
        eco = analyze_item_matrix(
            self.matrix, self.policies["ecologic"], router="ecologic",
            overhead=load_router_overhead("ecologic_keyword"),
            n_boot=0, verify=True,
        )
        self.assertEqual(eco["usd"]["status"], "dominated")
        self.assertTrue(eco["usd"]["routing_is_dominated"])
        self.assertLess(eco["usd"]["raw_router_savings"], 0.0)
        self.assertIsNone(eco["usd"]["percentage_of_break_even_used"])
        self.assertTrue(eco["usd"]["numeric_verified"])

        t2 = analyze_item_matrix(
            self.matrix, self.policies["always_t2"], router="always_t2",
            n_boot=0, verify=True,
        )
        self.assertEqual(t2["usd"]["status"], "neutral")
        self.assertAlmostEqual(t2["usd"]["raw_router_savings"], 0.0, places=12)
        self.assertAlmostEqual(t2["usd"]["percentage_of_break_even_used"], 0.0)

        oracle = analyze_item_matrix(
            self.matrix, self.policies["oracle_usd"], router="oracle_usd",
            n_boot=0, verify=True,
        )
        self.assertFalse(oracle["usd"]["same_quality_feasible"])
        self.assertTrue(math.isinf(oracle["usd"]["break_even_overhead"]))
        self.assertEqual(oracle["usd"]["status"], "beneficial")
        self.assertEqual(oracle["usd"]["percentage_of_break_even_used"], 0.0)
        self.assertGreater(oracle["usd"]["quality_advantage_at_router_cost"], 0.0)

    def test_table_and_save(self):
        payload = table_from_policies(
            self.matrix,
            {k: self.policies[k] for k in ("ecologic", "always_t2", "oracle_usd")},
            n_boot=0,
            verify=True,
        )
        self.assertEqual(payload["status_counts"]["usd"]["dominated"], 1)
        self.assertEqual(payload["status_counts"]["usd"]["neutral"], 1)
        self.assertEqual(payload["status_counts"]["usd"]["beneficial"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            # save_tables writes under results/; just check csv text here
            csv_text = rows_to_csv(payload["rows"])
            self.assertIn("ecologic", csv_text)
            self.assertIn("usd", csv_text)
            Path(tmp, "probe.csv").write_text(csv_text)


class TestNoFittingImports(unittest.TestCase):
    def test_breakeven_does_not_import_sklearn_or_torch(self):
        import woais_experiments.accounting.breakeven as be

        src = Path(be.__file__).read_text()
        for banned in ("sklearn", "torch", "xgboost", "GradientBoosting"):
            self.assertNotIn(banned, src)

    def test_raw_savings_bootstrap_uses_raw_replicates(self):
        import inspect
        import woais_experiments.accounting.breakeven as be
        src = inspect.getsource(be.analyze_assignment)
        self.assertIn("boot_raw_usd", src)
        compact = "".join(src.split())
        self.assertIn('_ci_payload(raw_s,row["raw_router_savings"]', compact)


if __name__ == "__main__":
    unittest.main()
