"""Regression tests for the adversarial woais_experiments audit fixes."""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.accounting.breakeven import analyze_assignment
from woais_experiments.accounting.costs import (
    cost_fn_energy,
    evaluate_assignment,
    paper_energy_rates,
)
from woais_experiments.frozen import matrix_from_calls, to_jsonable
from woais_experiments.routing.policies import always_tier, evaluate_policies
from woais_experiments.routing.robustness import (
    QUALITY_TARGETS,
    apply_latency_penalty,
    attach_bh,
    format_significance,
)
from woais_experiments.statistics.paired_tests import paired_permutation_test


class TestNoZeroFillMissing(unittest.TestCase):
    def test_incomplete_calls_are_dropped_not_zeroed(self):
        calls = []
        for item in ("a", "b"):
            for t in (1, 2, 3):
                calls.append({
                    "item_id": item,
                    "tier": t,
                    "correct": True,
                    "usd": 0.01 * t,
                    "latency_s": 0.5,
                    "total_tokens": 10,
                    "prompt_tokens": 6,
                    "completion_tokens": 4,
                    "benchmark": "x",
                })
        calls.append({
            "item_id": "ghost",
            "tier": 1,
            "correct": None,
            "usd": None,
            "latency_s": None,
            "total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        })
        matrix = matrix_from_calls(calls)
        self.assertEqual(matrix.n, 2)
        self.assertGreaterEqual(matrix.n_dropped_incomplete, 1)
        self.assertNotIn((1, "ghost"), matrix.usd)


class TestJsonInf(unittest.TestCase):
    def test_infinity_serializes_as_string_not_null(self):
        self.assertEqual(to_jsonable(math.inf), "Infinity")
        self.assertEqual(to_jsonable(float("-inf")), "-Infinity")
        import json
        text = json.dumps(to_jsonable({"x": math.inf}), allow_nan=False)
        self.assertIn("Infinity", text)
        self.assertNotIn("null", text)


class TestEnergyProvenance(unittest.TestCase):
    def test_energy_assignment_is_labeled_modelled(self):
        calls = []
        for item in ("a", "b"):
            for t in (1, 2, 3):
                calls.append({
                    "item_id": item, "tier": t, "correct": True,
                    "usd": 0.01, "latency_s": 0.2, "total_tokens": 1000,
                    "prompt_tokens": 500, "completion_tokens": 500, "benchmark": "x",
                })
        matrix = matrix_from_calls(calls)
        stats = evaluate_assignment(
            matrix, always_tier(matrix.item_ids, 2), cost_fn_energy(matrix),
            name="always_t2",
            cost_kind="modelled_energy",
            cost_unit="J",
            cost_provenance="tokens/1000 * paper_energy_per_1k (not metered)",
        )
        self.assertEqual(stats["cost_kind"], "modelled_energy")
        self.assertEqual(stats["cost_unit"], "J")
        self.assertIn("not metered", stats["cost_provenance"])

    def test_paper_energy_rates_do_not_import_api(self):
        src = inspect.getsource(paper_energy_rates)
        self.assertNotIn("from api import", src)
        rates = paper_energy_rates()
        self.assertEqual(rates[1], 0.5)


class TestMcnemarBh(unittest.TestCase):
    def test_policy_mcnemar_family_has_adjusted_p_and_effect_size(self):
        calls = []
        for i, item in enumerate(("a", "b", "c", "d")):
            for t in (1, 2, 3):
                calls.append({
                    "item_id": item, "tier": t,
                    "correct": True if t == 2 or i == 0 else False,
                    "usd": 0.01 * t, "latency_s": 0.2, "total_tokens": 10,
                    "prompt_tokens": 5, "completion_tokens": 5, "benchmark": "x",
                })
        matrix = matrix_from_calls(calls)
        policies = {
            "ecologic": always_tier(matrix.item_ids, 1),
            "always_t2": always_tier(matrix.item_ids, 2),
        }
        out = evaluate_policies(matrix, policies, lambda t, i: matrix.usd[(t, i)])
        row = out["mcnemar_vs"]["always_t2"]
        self.assertIn("p_adjusted", row)
        self.assertIn("effect_size", row)
        self.assertEqual(row["p_raw"], row["p_value"])


class TestRobustnessStats(unittest.TestCase):
    def test_bh_rewrites_significance_with_effect_size(self):
        rows = [
            {
                "setting": "baseline", "p_raw": 0.01, "effect_size": 0.4,
                "significance": format_significance(0.01),
            },
            {
                "setting": "model_pricing", "p_raw": 0.04, "effect_size": 0.2,
                "significance": format_significance(0.04),
            },
            {
                "setting": "bootstrap_resample", "p_raw": None, "effect_size": None,
                "significance": "n/a",
            },
        ]
        attach_bh(rows)
        self.assertIsNotNone(rows[0]["p_adjusted"])
        self.assertIn("p_adj", rows[0]["significance"])
        self.assertIn("dz=", rows[0]["significance"])
        self.assertIn("bootstrap", rows[2]["significance"])

    def test_latency_penalty_refuses_to_impute_zero(self):
        cost = np.ones((2, 2))
        lat = np.array([[1.0, np.nan], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            apply_latency_penalty(cost, lat, 0.01)

    def test_quality_grid_is_not_eval_peek(self):
        self.assertNotIn(0.868, QUALITY_TARGETS)
        self.assertNotIn(0.923, QUALITY_TARGETS)


class TestPermutationDenominator(unittest.TestCase):
    def test_nonfinite_nulls_are_not_in_the_p_denominator(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.5, 1.0])
        got = paired_permutation_test(a, b, "mean", n_perm=50, seed=1)
        self.assertTrue(got["available"])
        self.assertIn("n_nonfinite_nulls", got)
        self.assertEqual(got["n_perm"], 8)
        self.assertEqual(got["n_nonfinite_nulls"], 0)

    def test_nonfinite_cohens_dz_uses_finite_denominator(self):
        a = np.array([2.0, 0.0])
        b = np.array([1.0, 1.0])
        got = paired_permutation_test(a, b, "cohens_dz", n_perm=50, seed=1)
        self.assertGreater(got["n_nonfinite_nulls"], 0)
        self.assertEqual(got["n_finite_nulls"] + got["n_nonfinite_nulls"], got["n_perm"])
        self.assertLessEqual(got["p_raw"], 1.0)


class TestRouterOverheadInHeadline(unittest.TestCase):
    def test_overhead_is_added_only_for_ecologic(self):
        calls = []
        for item in ("a", "b"):
            for t in (1, 2, 3):
                calls.append({
                    "item_id": item, "tier": t, "correct": True,
                    "usd": 1.0, "latency_s": 0.2, "total_tokens": 10,
                    "prompt_tokens": 5, "completion_tokens": 5, "benchmark": "x",
                })
        matrix = matrix_from_calls(calls)
        policies = {
            "ecologic": always_tier(matrix.item_ids, 1),
            "always_t2": always_tier(matrix.item_ids, 2),
        }
        out = evaluate_policies(
            matrix, policies, lambda t, i: 1.0, overhead_per_query=0.25,
        )
        self.assertAlmostEqual(out["policies"]["ecologic"]["router_overhead_per_query"], 0.25)
        self.assertAlmostEqual(out["policies"]["always_t2"]["router_overhead_per_query"], 0.0)
        self.assertAlmostEqual(
            out["policies"]["ecologic"]["cost"],
            out["policies"]["ecologic"]["inference_cost"] + 0.25 * matrix.n,
        )


class TestPathsAndInfWiring(unittest.TestCase):
    def test_analyze_assignment_collects_raw_bootstrap_list(self):
        src = inspect.getsource(analyze_assignment)
        self.assertIn("boot_raw_usd", src)
        compact = "".join(src.split())
        self.assertIn('_ci_payload(raw_s,row["raw_router_savings"]', compact)
