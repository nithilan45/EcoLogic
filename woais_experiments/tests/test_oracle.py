"""Exact vs approximate MCKP oracles on analytically solvable instances."""

from __future__ import annotations

import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.routing.mckp import (
    MCKPInstance,
    approximate_solve,
    dp_solve,
    enumerate_solve,
    exact_solve,
    greedy_solve,
    lp_relaxation,
    milp_solve,
    solve,
    unconstrained_oracle,
)
from woais_experiments.routing.oracle import (
    budget_sweep,
    captured_fraction,
    evaluate_at_budget,
    instance_from_arrays,
    sweep_to_csv,
)


def _two_query_line() -> MCKPInstance:
    # Q1: A=(1,1), B=(3,1).  Q2: A=(1,0), B=(3,1).
    return instance_from_arrays(
        cost=np.array([[1.0, 3.0], [1.0, 3.0]]),
        quality=np.array([[1.0, 1.0], [0.0, 1.0]]),
        names=("A", "B"),
        query_ids=("q1", "q2"),
    )


def _greedy_gap() -> MCKPInstance:
    # Budget 3: greedy takes class-0's efficiency-1 upgrade (cost 2, q=2).
    # Exact takes class-1's upgrade (cost 3, q=2.2).
    return instance_from_arrays(
        cost=np.array([[0.0, 2.0, 5.0], [0.0, 3.0, 9.0]]),
        quality=np.array([[0.0, 2.0, 2.05], [0.0, 2.2, 0.0]]),
        names=("cheap", "mid", "far"),
    )


class TestTwoQueryOracle(unittest.TestCase):
    def setUp(self):
        self.inst = _two_query_line()

    def test_min_budget_is_always_cheap(self):
        sol = exact_solve(self.inst, 2.0)
        self.assertTrue(sol.feasible)
        self.assertAlmostEqual(sol.total_cost, 2.0)
        self.assertAlmostEqual(sol.mean_quality, 0.5)
        np.testing.assert_array_equal(sol.choice, [0, 0])

    def test_budget_4_picks_the_valuable_upgrade(self):
        sol = exact_solve(self.inst, 4.0)
        self.assertAlmostEqual(sol.total_cost, 4.0)
        self.assertAlmostEqual(sol.mean_quality, 1.0)
        np.testing.assert_array_equal(sol.choice, [0, 1])

    def test_budget_3_cannot_upgrade(self):
        sol = exact_solve(self.inst, 3.0)
        self.assertAlmostEqual(sol.mean_quality, 0.5)

    def test_enumerate_matches_dfs(self):
        for b in (2, 3, 4, 6):
            a = exact_solve(self.inst, b)
            e = enumerate_solve(self.inst, b)
            self.assertAlmostEqual(a.total_quality, e.total_quality)
            self.assertAlmostEqual(a.total_cost, e.total_cost)

    def test_dp_matches_exact(self):
        for b in (2, 3, 4, 6):
            d = dp_solve(self.inst, b)
            x = exact_solve(self.inst, b)
            self.assertAlmostEqual(d.total_quality, x.total_quality)
            self.assertAlmostEqual(d.mean_quality, x.mean_quality)

    def test_greedy_matches_exact_on_this_instance(self):
        for b in (2, 4, 6):
            g = greedy_solve(self.inst, b)
            x = exact_solve(self.inst, b)
            self.assertAlmostEqual(g.total_quality, x.total_quality)

    def test_unconstrained_is_min_cost_max_quality(self):
        u = unconstrained_oracle(self.inst)
        self.assertAlmostEqual(u.mean_quality, 1.0)
        self.assertAlmostEqual(u.total_cost, 4.0)


class TestGapsAtMatchedBudget(unittest.TestCase):
    def setUp(self):
        self.inst = _two_query_line()

    def test_perfect_router_captures_all_routing_value(self):
        # Oracle assignment itself: q1→A, q2→B, cost 4, quality 1.
        row = evaluate_at_budget(self.inst, 4.0, np.array([0, 1]), method="exact")
        self.assertTrue(row["router_feasible"])
        self.assertAlmostEqual(row["learned_router_quality"], 1.0)
        self.assertAlmostEqual(row["oracle_quality"], 1.0)
        # Static at mean cost 2: 50-50 A/B → quality 0.75.
        self.assertAlmostEqual(row["static_frontier_quality"], 0.75)
        self.assertAlmostEqual(row["router_vs_static_gap"], 0.25)
        self.assertAlmostEqual(row["oracle_vs_router_gap"], 0.0)
        self.assertAlmostEqual(row["fraction_of_available_routing_value_captured"], 1.0)

    def test_always_cheap_router_captures_none_at_high_budget(self):
        row = evaluate_at_budget(self.inst, 4.0, np.array([0, 0]), method="exact")
        self.assertAlmostEqual(row["learned_router_quality"], 0.5)
        self.assertAlmostEqual(row["static_frontier_quality"], 0.75)
        self.assertAlmostEqual(row["oracle_quality"], 1.0)
        self.assertAlmostEqual(row["router_vs_static_gap"], -0.25)
        self.assertAlmostEqual(row["oracle_vs_router_gap"], 0.5)
        self.assertAlmostEqual(row["fraction_of_available_routing_value_captured"], -1.0)

    def test_router_infeasible_below_its_cost(self):
        row = evaluate_at_budget(self.inst, 3.0, np.array([0, 1]), method="exact")
        self.assertFalse(row["router_feasible"])
        self.assertIsNone(row["router_vs_static_gap"])
        self.assertAlmostEqual(row["oracle_quality"], 0.5)

    def test_captured_fraction_undefined_when_no_headroom(self):
        self.assertIsNone(captured_fraction(0.5, 0.5, 0.5))
        self.assertAlmostEqual(captured_fraction(0.8, 0.5, 1.0), 0.6)


class TestGreedyCanBeSuboptimal(unittest.TestCase):
    def test_exact_beats_greedy(self):
        inst = _greedy_gap()
        g = greedy_solve(inst, 3.0)
        x = exact_solve(inst, 3.0)
        e = enumerate_solve(inst, 3.0)
        self.assertAlmostEqual(x.total_quality, 2.2)
        self.assertAlmostEqual(e.total_quality, 2.2)
        self.assertLess(g.total_quality + 1e-12, x.total_quality)
        self.assertAlmostEqual(g.total_quality, 2.0)

    def test_dp_matches_exact_here(self):
        inst = _greedy_gap()
        d = dp_solve(inst, 3.0)
        self.assertAlmostEqual(d.total_quality, 2.2)

    def test_approx_recovers_via_single_class_jump(self):
        inst = _greedy_gap()
        a = approximate_solve(inst, 3.0)
        self.assertAlmostEqual(a.total_quality, 2.2)

    def test_lp_is_an_upper_bound(self):
        inst = _greedy_gap()
        lp = lp_relaxation(inst, 3.0)
        x = exact_solve(inst, 3.0)
        self.assertGreaterEqual(lp.total_quality + 1e-9, x.total_quality)


class TestRandomSmallEqualsExact(unittest.TestCase):
    def test_dfs_enum_dp_and_milp_agree(self):
        rng = np.random.default_rng(20260909)
        for n in (3, 4, 5):
            cost = rng.integers(1, 8, size=(n, 3)).astype(float)
            quality = rng.integers(0, 5, size=(n, 3)).astype(float)
            inst = MCKPInstance(cost, quality)
            lo, hi = inst.min_total_cost(), inst.max_total_cost()
            for b in {lo, hi, float(int((lo + hi) // 2))}:
                exact = exact_solve(inst, b)
                enum = enumerate_solve(inst, b)
                dp = dp_solve(inst, b)
                greedy = greedy_solve(inst, b)
                approx = approximate_solve(inst, b)
                lp = lp_relaxation(inst, b)
                self.assertAlmostEqual(exact.total_quality, enum.total_quality, places=9)
                self.assertAlmostEqual(exact.total_quality, dp.total_quality, places=9)
                self.assertLessEqual(greedy.total_quality, exact.total_quality + 1e-9)
                self.assertLessEqual(approx.total_quality, exact.total_quality + 1e-9)
                self.assertGreaterEqual(lp.total_quality, exact.total_quality - 1e-9)
                try:
                    milp = milp_solve(inst, b)
                except ImportError:
                    continue
                self.assertAlmostEqual(milp.total_quality, exact.total_quality, places=6)

    def test_auto_uses_exact_when_small(self):
        inst = _two_query_line()
        sol = solve(inst, 4.0, method="auto")
        self.assertEqual(sol.method, "exact")
        self.assertAlmostEqual(sol.mean_quality, 1.0)


class TestInfeasibleBudget(unittest.TestCase):
    def test_below_cheapest(self):
        inst = _two_query_line()
        sol = exact_solve(inst, 1.0)
        self.assertFalse(sol.feasible)
        row = evaluate_at_budget(inst, 1.0, np.array([0, 0]), method="exact")
        self.assertFalse(row["oracle_feasible"])
        self.assertFalse(row["static_feasible"])


class TestBudgetSweep(unittest.TestCase):
    def test_rows_contain_three_qualities_and_gaps(self):
        inst = _two_query_line()
        payload = budget_sweep(
            inst,
            budgets=[2.0, 4.0, 6.0],
            router_choice=np.array([0, 1]),
            method="exact",
        )
        self.assertEqual(len(payload["sweep"]), 3)
        at4 = payload["sweep"][1]
        self.assertAlmostEqual(at4["static_frontier_quality"], 0.75)
        self.assertAlmostEqual(at4["learned_router_quality"], 1.0)
        self.assertAlmostEqual(at4["oracle_quality"], 1.0)
        self.assertAlmostEqual(at4["fraction_of_available_routing_value_captured"], 1.0)
        gaps = payload["gaps_at_router_cost"]
        self.assertAlmostEqual(gaps["budget"], 4.0)
        csv_text = sweep_to_csv(payload)
        self.assertIn("oracle_quality", csv_text)
        self.assertIn("fraction_of_available_routing_value_captured", csv_text)

    def test_default_grid_includes_router_cost(self):
        inst = _two_query_line()
        payload = budget_sweep(inst, router_choice=np.array([0, 1]), method="exact", n_grid=5)
        costs = [row["budget"] for row in payload["sweep"]]
        self.assertTrue(any(abs(c - 4.0) < 1e-9 for c in costs))


class TestScaledDP(unittest.TestCase):
    def test_half_integer_costs_with_scale(self):
        inst = instance_from_arrays(
            cost=np.array([[0.5, 1.5], [0.5, 1.5]]),
            quality=np.array([[1.0, 1.0], [0.0, 1.0]]),
        )
        d = dp_solve(inst, 2.0, scale=2.0)
        x = exact_solve(inst, 2.0)
        self.assertAlmostEqual(d.total_quality, x.total_quality)


if __name__ == "__main__":
    unittest.main()
