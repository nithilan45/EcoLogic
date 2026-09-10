"""Analytically solvable tests for query-independent static baselines. No ML."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.routing.cost_matching import (
    compare_router_to_static,
    match_cost,
    match_quality,
    router_means,
)
from woais_experiments.routing.frontier import (
    StaticMarket,
    interpolate_at_cost,
    interpolate_at_quality,
    pairwise_pareto_filter,
    upper_hull_indices,
)
from woais_experiments.routing.static_baselines import (
    always_cheapest,
    always_each_model,
    always_most_expensive,
    catalog,
    evaluate_assignment_against_static,
    general_mixture,
    market_from_means,
    market_from_panel,
    max_quality_for_cost,
    min_cost_for_quality,
    pairwise_random_mixtures,
    pairwise_segments,
    uniform_mixture,
)


def _slopes(market: StaticMarket) -> list[float]:
    idx = upper_hull_indices(market)
    out = []
    for a, b in zip(idx, idx[1:]):
        dc = market.mean_cost[b] - market.mean_cost[a]
        dq = market.mean_quality[b] - market.mean_quality[a]
        out.append(dq / dc)
    return out


class TestTwoModelLine(unittest.TestCase):
    """A=(C=1, q=0.5), B=(C=3, q=0.9). Mix at cost 2 is f=1/2, q=0.7."""

    def setUp(self):
        self.m = StaticMarket(["A", "B"], [1.0, 3.0], [0.5, 0.9])

    def test_midpoint_quality(self):
        mix = interpolate_at_cost(self.m, 2.0, mode="equal")
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 2.0)
        self.assertAlmostEqual(mix.quality, 0.7)
        self.assertAlmostEqual(mix.weights["A"], 0.5)
        self.assertAlmostEqual(mix.weights["B"], 0.5)

    def test_endpoints(self):
        lo = interpolate_at_cost(self.m, 1.0)
        hi = interpolate_at_cost(self.m, 3.0)
        self.assertAlmostEqual(lo.quality, 0.5)
        self.assertAlmostEqual(hi.quality, 0.9)
        self.assertAlmostEqual(lo.weights["A"], 1.0)
        self.assertAlmostEqual(hi.weights["B"], 1.0)

    def test_min_cost_at_quality_0_7(self):
        mix = interpolate_at_quality(self.m, 0.7)
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 2.0)
        self.assertAlmostEqual(mix.quality, 0.7)

    def test_quality_below_cheap_uses_cheap(self):
        mix = interpolate_at_quality(self.m, 0.4)
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 1.0)
        self.assertAlmostEqual(mix.quality, 0.5)

    def test_always_cheapest_and_expensive(self):
        self.assertEqual(list(always_cheapest(self.m).weights), ["A"])
        self.assertEqual(list(always_most_expensive(self.m).weights), ["B"])


class TestThreeModelChord(unittest.TestCase):
    """A=(1,0.4), B=(2,0.8), C=(4,0.9). At cost 3, mix B/C with t=1/2 → q=0.85."""

    def setUp(self):
        self.m = StaticMarket(["A", "B", "C"], [1.0, 2.0, 4.0], [0.4, 0.8, 0.9])

    def test_hull_keeps_all_three(self):
        names = [self.m.names[i] for i in upper_hull_indices(self.m)]
        self.assertEqual(names, ["A", "B", "C"])
        s = _slopes(self.m)
        self.assertGreater(s[0], s[1])  # 0.4 > 0.05

    def test_cost_3_quality_0_85(self):
        mix = interpolate_at_cost(self.m, 3.0)
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 3.0)
        self.assertAlmostEqual(mix.quality, 0.85)
        self.assertAlmostEqual(mix.weights["B"], 0.5)
        self.assertAlmostEqual(mix.weights["C"], 0.5)
        self.assertNotIn("A", mix.weights)

    def test_cost_1_5_on_first_edge(self):
        mix = interpolate_at_cost(self.m, 1.5)
        self.assertAlmostEqual(mix.quality, 0.6)
        self.assertAlmostEqual(mix.weights["A"], 0.5)
        self.assertAlmostEqual(mix.weights["B"], 0.5)

    def test_min_cost_for_quality_0_85(self):
        mix = min_cost_for_quality(self.m, 0.85)
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 3.0)
        self.assertAlmostEqual(mix.quality, 0.85)

    def test_uniform_is_barycenter(self):
        u = uniform_mixture(self.m)
        self.assertAlmostEqual(u.cost, (1 + 2 + 4) / 3)
        self.assertAlmostEqual(u.quality, (0.4 + 0.8 + 0.9) / 3)
        self.assertAlmostEqual(sum(u.weights.values()), 1.0)


class TestMixtureDominatedInterior(unittest.TestCase):
    """B sits below chord AC: pairwise undominated, not a hull vertex."""

    def setUp(self):
        # chord AC at cost 2: 0.4 + 1/3 * 0.5 = 0.4 + 1/6 ≈ 0.5667; B=0.55 is below
        self.m = StaticMarket(["A", "B", "C"], [1.0, 2.0, 4.0], [0.4, 0.55, 0.9])

    def test_b_pairwise_kept_hull_dropped(self):
        pairwise = [self.m.names[i] for i in pairwise_pareto_filter(self.m)]
        hull = [self.m.names[i] for i in upper_hull_indices(self.m)]
        self.assertEqual(pairwise, ["A", "B", "C"])
        self.assertEqual(hull, ["A", "C"])

    def test_cost_2_uses_ac_not_b(self):
        mix = interpolate_at_cost(self.m, 2.0)
        self.assertAlmostEqual(mix.quality, 0.4 + (1.0 / 3.0) * 0.5)
        self.assertIn("A", mix.weights)
        self.assertIn("C", mix.weights)
        self.assertNotIn("B", mix.weights)
        self.assertGreater(mix.quality, 0.55)


class TestCollinearDrop(unittest.TestCase):
    def test_middle_collinear_vertex_dropped(self):
        m = StaticMarket(["A", "B", "C"], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        hull = [m.names[i] for i in upper_hull_indices(m)]
        self.assertEqual(hull, ["A", "C"])
        mix = interpolate_at_cost(m, 1.0)
        self.assertAlmostEqual(mix.quality, 1.0)
        self.assertNotIn("B", mix.weights)


class TestDominatedExpensiveWorse(unittest.TestCase):
    def test_expensive_worse_is_not_on_hull(self):
        m = StaticMarket(
            ["cheap", "mid", "lemon"],
            [1.0, 2.0, 5.0],
            [0.4, 0.8, 0.3],
        )
        hull = [m.names[i] for i in upper_hull_indices(m)]
        self.assertEqual(hull, ["cheap", "mid"])
        pairwise = [m.names[i] for i in pairwise_pareto_filter(m)]
        self.assertNotIn("lemon", pairwise)

    def test_cost_above_hull_exact_is_infeasible(self):
        m = StaticMarket(["cheap", "mid", "lemon"], [1.0, 2.0, 5.0], [0.4, 0.8, 0.3])
        eq = interpolate_at_cost(m, 4.0, mode="equal")
        self.assertFalse(eq.feasible)
        cap = interpolate_at_cost(m, 4.0, mode="at_most")
        self.assertTrue(cap.feasible)
        self.assertAlmostEqual(cap.quality, 0.8)
        self.assertAlmostEqual(cap.cost, 2.0)


class TestEcoLogicLikeCheapestIsBest(unittest.TestCase):
    """T2 cheaper and more accurate: extra budget is not spent under at_most."""

    def setUp(self):
        self.m = StaticMarket(["T1", "T2", "T3"], [3.0, 1.0, 10.0], [0.80, 0.92, 0.90])

    def test_hull_is_only_t2(self):
        hull = [self.m.names[i] for i in upper_hull_indices(self.m)]
        self.assertEqual(hull, ["T2"])

    def test_at_most_does_not_spend_extra(self):
        mix = max_quality_for_cost(self.m, 5.0, mode="at_most")
        self.assertTrue(mix.feasible)
        self.assertAlmostEqual(mix.cost, 1.0)
        self.assertAlmostEqual(mix.quality, 0.92)
        self.assertIn("extra budget is not spent", mix.note)

    def test_equal_cost_5_infeasible(self):
        mix = max_quality_for_cost(self.m, 5.0, mode="equal")
        self.assertFalse(mix.feasible)

    def test_always_cheapest_is_t2(self):
        self.assertEqual(list(always_cheapest(self.m).weights), ["T2"])
        self.assertEqual(list(always_most_expensive(self.m).weights), ["T3"])


class TestSingleModel(unittest.TestCase):
    def test_everything_is_the_one_model(self):
        m = StaticMarket(["only"], [2.5], [0.7])
        self.assertEqual(upper_hull_indices(m), [0])
        self.assertTrue(interpolate_at_cost(m, 2.5).feasible)
        self.assertFalse(interpolate_at_cost(m, 3.0).feasible)
        q = interpolate_at_quality(m, 0.7)
        self.assertTrue(q.feasible)
        self.assertAlmostEqual(q.cost, 2.5)
        self.assertFalse(interpolate_at_quality(m, 0.71).feasible)
        self.assertEqual(catalog(m)["n_models"], 1)
        self.assertEqual(pairwise_random_mixtures(m), [])


class TestInfeasibleQuality(unittest.TestCase):
    def test_above_max_quality(self):
        m = StaticMarket(["A", "B"], [1.0, 3.0], [0.5, 0.9])
        mix = interpolate_at_quality(m, 0.95)
        self.assertFalse(mix.feasible)
        self.assertEqual(mix.kind, "quality_infeasible")


class TestCatalogAndGeneralMixture(unittest.TestCase):
    def setUp(self):
        self.m = StaticMarket(["A", "B", "C"], [1.0, 2.0, 4.0], [0.4, 0.8, 0.9])

    def test_always_each(self):
        each = always_each_model(self.m)
        self.assertEqual(set(each), {"A", "B", "C"})
        self.assertAlmostEqual(each["B"].cost, 2.0)
        self.assertAlmostEqual(each["B"].quality, 0.8)

    def test_three_pairwise_50_50(self):
        pairs = pairwise_random_mixtures(self.m, f=0.5)
        self.assertEqual(len(pairs), 3)
        ab = next(p for p in pairs if set(p.weights) == {"A", "B"})
        self.assertAlmostEqual(ab.cost, 1.5)
        self.assertAlmostEqual(ab.quality, 0.6)

    def test_pairwise_grid_includes_endpoints(self):
        segs = pairwise_segments(self.m, n_grid=3)
        self.assertEqual(len(segs), 3 * 3)

    def test_general_simplex_weights(self):
        mix = general_mixture(self.m, {"A": 0.2, "B": 0.3, "C": 0.5})
        self.assertAlmostEqual(mix.cost, 0.2 * 1 + 0.3 * 2 + 0.5 * 4)
        self.assertAlmostEqual(mix.quality, 0.2 * 0.4 + 0.3 * 0.8 + 0.5 * 0.9)
        self.assertAlmostEqual(sum(mix.weights.values()), 1.0)

    def test_weights_are_normalized(self):
        mix = general_mixture(self.m, {"A": 1.0, "B": 1.0})
        self.assertAlmostEqual(mix.weights["A"], 0.5)
        self.assertAlmostEqual(mix.weights["B"], 0.5)

    def test_catalog_keys(self):
        cat = catalog(self.m)
        self.assertEqual(cat["always_cheapest"]["weights"], {"A": 1.0})
        self.assertEqual(cat["always_most_expensive"]["weights"], {"C": 1.0})
        self.assertEqual(len(cat["pairwise_random"]), 3)
        self.assertEqual(cat["hull_model_names"], ["A", "B", "C"])

    def test_tie_cheapest_prefers_higher_quality(self):
        m = StaticMarket(["low", "high"], [1.0, 1.0], [0.1, 0.9])
        self.assertEqual(list(always_cheapest(m).weights), ["high"])

    def test_tie_expensive_prefers_lower_quality(self):
        m = StaticMarket(["good", "bad"], [5.0, 5.0], [0.9, 0.1])
        self.assertEqual(list(always_most_expensive(m).weights), ["bad"])


class TestMarketFromPanel(unittest.TestCase):
    def test_unconditional_means(self):
        names = ["A", "B"]
        cost = np.array([[1.0, 5.0], [3.0, 5.0]])
        quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        m = market_from_panel(names, cost, quality)
        self.assertAlmostEqual(m.mean_cost[0], 2.0)
        self.assertAlmostEqual(m.mean_cost[1], 5.0)
        self.assertAlmostEqual(m.mean_quality[0], 0.5)
        self.assertAlmostEqual(m.mean_quality[1], 1.0)
        m2 = market_from_means(names, [2.0, 5.0], [0.5, 1.0])
        np.testing.assert_allclose(m.mean_cost, m2.mean_cost)


class TestCostMatchedRouter(unittest.TestCase):
    """Two queries, two models. Analytic router vs static at equal realized cost."""

    def setUp(self):
        self.names = ["A", "B"]
        self.cost = np.array([[1.0, 5.0], [3.0, 5.0]])
        self.quality = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.ids = ["q1", "q2"]
        self.market = market_from_panel(self.names, self.cost, self.quality)
        # A: (2.0, 0.5), B: (5.0, 1.0)

    def test_router_means_oracle(self):
        # q1→A (1,1), q2→B (5,1) → cost 3, quality 1
        assign = {"q1": "A", "q2": "B"}
        q, c = router_means(assign, self.names, self.cost, self.quality, query_ids=self.ids)
        self.assertAlmostEqual(q, 1.0)
        self.assertAlmostEqual(c, 3.0)

    def test_query_dependent_oracle_beats_static(self):
        assign = {"q1": "A", "q2": "B"}
        q, c = router_means(assign, self.names, self.cost, self.quality, query_ids=self.ids)
        report = compare_router_to_static(self.market, q, c)
        self.assertAlmostEqual(report["router_quality"], 1.0)
        self.assertAlmostEqual(report["router_realized_cost"], 3.0)
        # static at cost 3: t=(3-2)/(5-2)=1/3 on B → q = (2/3)*0.5 + (1/3)*1 = 2/3
        self.assertAlmostEqual(report["static_quality_at_same_cost"], 2.0 / 3.0)
        self.assertAlmostEqual(report["quality_advantage"], 1.0 - 2.0 / 3.0)
        self.assertAlmostEqual(report["relative_regret"], (2.0 / 3.0 - 1.0) / 0.5)
        self.assertLess(report["relative_regret"], 0.0)
        self.assertGreater(report["quality_advantage"], 0.0)
        # same quality 1.0: static must use always-B at cost 5
        self.assertAlmostEqual(report["static_cost_at_same_quality"], 5.0)
        self.assertAlmostEqual(report["router_cost_at_same_quality"], 3.0)
        self.assertAlmostEqual(report["cost_savings"], 2.0)

    def test_anti_oracle_loses_to_static(self):
        # q1→B (5,1), q2→A (3,0) → cost 4, quality 0.5
        assign = {"q1": "B", "q2": "A"}
        q, c = router_means(assign, self.names, self.cost, self.quality, query_ids=self.ids)
        report = compare_router_to_static(self.market, q, c)
        self.assertAlmostEqual(c, 4.0)
        self.assertAlmostEqual(q, 0.5)
        # static at 4: t=(4-2)/3 = 2/3 on B → q = (1/3)*0.5 + (2/3)*1 = 5/6
        self.assertAlmostEqual(report["static_quality_at_same_cost"], 5.0 / 6.0)
        self.assertAlmostEqual(report["quality_advantage"], 0.5 - 5.0 / 6.0)
        self.assertLess(report["quality_advantage"], 0.0)
        self.assertGreater(report["relative_regret"], 0.0)
        # quality 0.5 is already met by always-A at cost 2
        self.assertAlmostEqual(report["static_cost_at_same_quality"], 2.0)
        self.assertAlmostEqual(report["cost_savings"], 2.0 - 4.0)

    def test_evaluate_assignment_bundles_catalog(self):
        report = evaluate_assignment_against_static(
            self.names, self.cost, self.quality, {"q1": "A", "q2": "B"}, self.ids
        )
        self.assertIn("catalog", report)
        self.assertAlmostEqual(report["quality_advantage"], 1.0 / 3.0)
        self.assertEqual(report["catalog"]["n_models"], 2)

    def test_always_a_matches_itself(self):
        assign = {"q1": "A", "q2": "A"}
        q, c = router_means(assign, self.names, self.cost, self.quality, query_ids=self.ids)
        report = compare_router_to_static(self.market, q, c)
        self.assertAlmostEqual(report["quality_advantage"], 0.0)
        self.assertAlmostEqual(report["relative_regret"], 0.0)
        self.assertAlmostEqual(report["cost_savings"], 0.0)

    def test_match_wrappers(self):
        self.assertAlmostEqual(match_cost(self.market, 3.0).quality, 2.0 / 3.0)
        self.assertAlmostEqual(match_quality(self.market, 1.0).cost, 5.0)


class TestBudgetBelowCheapest(unittest.TestCase):
    def test_infeasible(self):
        m = StaticMarket(["A", "B"], [1.0, 3.0], [0.5, 0.9])
        mix = interpolate_at_cost(m, 0.5, mode="at_most")
        self.assertFalse(mix.feasible)
        eq = interpolate_at_cost(m, 0.5, mode="equal")
        self.assertFalse(eq.feasible)


class TestSameCostDuplicateModels(unittest.TestCase):
    def test_keeps_higher_quality_at_tied_cost(self):
        m = StaticMarket(["weak", "strong"], [2.0, 2.0], [0.4, 0.9])
        hull = [m.names[i] for i in upper_hull_indices(m)]
        self.assertEqual(hull, ["strong"])
        self.assertEqual(list(always_cheapest(m).weights), ["strong"])


class TestNoFittingImports(unittest.TestCase):
    def test_modules_do_not_import_sklearn_or_torch(self):
        import woais_experiments.routing.cost_matching as cm
        import woais_experiments.routing.frontier as fr
        import woais_experiments.routing.static_baselines as sb

        for mod in (cm, fr, sb):
            src = Path(mod.__file__).read_text()
            for banned in ("sklearn", "torch", "xgboost", "GradientBoosting", "fit("):
                self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main()
