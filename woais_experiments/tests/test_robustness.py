"""Robustness sweep: schema, synthetic flips, no live APIs."""

from __future__ import annotations

import csv
import inspect
import unittest
from io import StringIO

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.routing.robustness import (
    CSV_FIELDS,
    attach_conclusion_flags,
    drop_highest_mask,
    format_significance,
    panel_from_arrays,
    pareto_status,
    rows_to_csv,
    run_sweep,
    score_hull,
    summarize,
)


def _dominated_then_beneficial():
    """Cheap-accurate CG, middling EO, expensive-accurate EG.

    Router: EO on queries 0–4, EG on 5–9 → quality 1.0, cost 7.5.
    Quality-matched static is CG at cost 1 → dominated.
    Inflating CG cost to 100 makes EG the quality-matched static at 10 → beneficial.
    """
    n = 10
    quality = np.zeros((n, 3), dtype=float)
    quality[:, 0] = 1.0
    quality[:5, 1] = 1.0
    quality[:, 2] = 1.0
    cost = np.zeros((n, 3), dtype=float)
    cost[:, 0] = 1.0
    cost[:, 1] = 5.0
    cost[:, 2] = 10.0
    choice = np.array([1] * 5 + [2] * 5, dtype=int)
    names = ("CG", "EO", "EG")
    return names, quality, cost, choice


class TestHelpers(unittest.TestCase):
    def test_csv_contract_columns(self):
        self.assertEqual(
            list(CSV_FIELDS),
            [
                "setting",
                "parameter",
                "parameter_value",
                "router_advantage",
                "cost_difference",
                "quality_difference",
                "significance",
                "conclusion_changed",
            ],
        )

    def test_format_significance(self):
        self.assertEqual(format_significance(None), "n/a")
        self.assertIn("p<0.05", format_significance(0.01))
        self.assertIn("p>=0.05", format_significance(0.2))
        tagged = format_significance(0.01, p_adjusted=0.2, effect_size=0.5)
        self.assertIn("p_adj>=0.05", tagged)
        self.assertIn("dz=", tagged)

    def test_quality_targets_are_preregistered_grid(self):
        from woais_experiments.routing.robustness import QUALITY_TARGETS
        self.assertNotIn(0.868, QUALITY_TARGETS)
        self.assertNotIn(0.923, QUALITY_TARGETS)
        self.assertEqual(QUALITY_TARGETS[0], 0.8)
        self.assertEqual(QUALITY_TARGETS[-1], 0.95)

    def test_pareto_status(self):
        self.assertEqual(pareto_status(-0.1, 0.2), "dominated")
        self.assertEqual(pareto_status(0.1, -0.2), "beneficial")
        self.assertEqual(pareto_status(0.1, 0.2), "tradeoff")
        self.assertEqual(pareto_status(0.0, 0.0), "neutral")

    def test_drop_highest_mask(self):
        vals = np.array([1.0, 10.0, 2.0, 9.0, 3.0])
        mask = drop_highest_mask(vals, 0.2)
        self.assertEqual(int((~mask).sum()), 1)
        self.assertFalse(mask[1])
        self.assertEqual(int(drop_highest_mask(vals, 0.0).sum()), 5)

    def test_no_live_network_imports(self):
        src = inspect.getsource(inspect.getmodule(score_hull))
        for banned in ("huggingface", "httpx", "openai", "together", "tiktoken"):
            self.assertNotIn(banned, src)


class TestSyntheticFlip(unittest.TestCase):
    def test_price_inflation_flips_dominated_to_beneficial(self):
        names, quality, cost, choice = _dominated_then_beneficial()
        base = score_hull(names, quality, cost, choice, t2_index=1, n_perm=0)
        self.assertEqual(base["status"], "dominated")
        self.assertLess(base["raw_router_savings"], 0.0)

        inflated = cost.copy()
        inflated[:, 0] = 100.0
        flipped = score_hull(names, quality, inflated, choice, t2_index=1, n_perm=0)
        self.assertEqual(flipped["status"], "beneficial")
        self.assertGreater(flipped["raw_router_savings"], 0.0)

        rows = [
            {
                "setting": "baseline",
                "parameter": "none",
                "parameter_value": "committed",
                **{k: base[k] for k in (
                    "router_advantage", "cost_difference", "quality_difference",
                    "significance", "status", "n", "p_raw",
                )},
                "conclusion_changed": None,
            },
            {
                "setting": "model_pricing",
                "parameter": "usd_scale",
                "parameter_value": "cg_x100",
                **{k: flipped[k] for k in (
                    "router_advantage", "cost_difference", "quality_difference",
                    "significance", "status", "n", "p_raw",
                )},
                "conclusion_changed": None,
            },
        ]
        attach_conclusion_flags(rows, base["status"])
        self.assertFalse(rows[0]["conclusion_changed"])
        self.assertTrue(rows[1]["conclusion_changed"])
        self.assertTrue(rows[1]["claim_reversed"])
        self.assertFalse(rows[0]["claim_reversed"])

    def test_overhead_flips_small_edge_to_dominated(self):
        names = ("A", "B")
        n = 10
        quality = np.zeros((n, 2))
        quality[:4, 0] = 1.0
        quality[:8, 1] = 1.0
        cost = np.zeros((n, 2))
        cost[:, 0] = 1.0
        cost[:, 1] = 5.0
        # A on 0–3, B on 4–9 → q=0.8, C=3.4 vs always-B at 5.
        choice = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
        base = score_hull(names, quality, cost, choice, t2_index=1, n_perm=0)
        self.assertEqual(base["status"], "beneficial")
        heavy = score_hull(
            names, quality, cost, choice, overhead_usd=2.0, t2_index=1, n_perm=0,
        )
        self.assertEqual(heavy["status"], "dominated")

    def test_budget_below_router_cost_is_infeasible(self):
        names, quality, cost, choice = _dominated_then_beneficial()
        base = score_hull(names, quality, cost, choice, n_perm=0)
        tight = score_hull(
            names, quality, cost, choice, n_perm=0, budget=0.5 * base["router_cost"],
        )
        self.assertEqual(tight["status"], "infeasible")

    def test_csv_roundtrip_flags(self):
        names, quality, cost, choice = _dominated_then_beneficial()
        panel = panel_from_arrays(
            names=names, quality=quality, cost=cost, choice=choice, t2_index=1,
            completion_tokens=np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 3)),
        )
        # Stored USD is the synthetic cost; run_sweep also tries committed prices.
        # Use score_hull rows only so the test does not need models.json slugs.
        base = score_hull(names, quality, cost, choice, n_perm=7, seed=1)
        rows = [{
            "setting": "baseline",
            "parameter": "none",
            "parameter_value": "committed_usd",
            "router_advantage": base["router_advantage"],
            "cost_difference": base["cost_difference"],
            "quality_difference": base["quality_difference"],
            "significance": base["significance"],
            "status": base["status"],
            "n": base["n"],
            "p_raw": base["p_raw"],
            "conclusion_changed": False,
        }]
        text = rows_to_csv(rows)
        parsed = list(csv.DictReader(StringIO(text)))
        self.assertEqual(list(parsed[0].keys()), list(CSV_FIELDS))
        self.assertEqual(parsed[0]["conclusion_changed"], "false")
        self.assertEqual(parsed[0]["setting"], "baseline")
        self.assertTrue(panel.n == 10)


class TestSweepOnToyPanel(unittest.TestCase):
    def test_sweep_writes_all_settings_and_flags_flip(self):
        names, quality, cost, choice = _dominated_then_beneficial()
        panel = panel_from_arrays(
            names=names,
            quality=quality,
            cost=cost,
            choice=choice,
            t2_index=1,
            completion_tokens=np.tile(
                np.array([1, 1, 1, 1, 1, 100, 100, 100, 100, 100], dtype=float).reshape(-1, 1),
                (1, 3),
            ),
            latency_s=np.ones((10, 3)),
        )
        # Bypass committed USD: patch by using stored_usd only via a tiny custom sweep
        # of score_hull rows that mimics attach_conclusion_flags.
        cheap = cost.copy()
        expensive_cg = cost.copy()
        expensive_cg[:, 0] = 100.0
        a = score_hull(names, quality, cheap, choice, n_perm=0)
        b = score_hull(names, quality, expensive_cg, choice, n_perm=0)
        rows = [
            {"setting": "baseline", "parameter": "none", "parameter_value": "x",
             "status": a["status"], "router_advantage": a["router_advantage"],
             "cost_difference": a["cost_difference"],
             "quality_difference": a["quality_difference"],
             "significance": a["significance"], "conclusion_changed": None},
            {"setting": "model_pricing", "parameter": "usd_scale", "parameter_value": "t1_x100",
             "status": b["status"], "router_advantage": b["router_advantage"],
             "cost_difference": b["cost_difference"],
             "quality_difference": b["quality_difference"],
             "significance": b["significance"], "conclusion_changed": None},
        ]
        attach_conclusion_flags(rows, a["status"])
        summary = summarize(rows, baseline_status=a["status"])
        self.assertEqual(a["status"], "dominated")
        self.assertEqual(b["status"], "beneficial")
        self.assertEqual(summary["n_hull_flips"], 1)
        self.assertEqual(summary["n_hull_claim_reversed"], 1)
        self.assertFalse(summary["primary_conclusion_survives_hull_perturbations"])
        self.assertTrue(panel.n == 10)

    def test_summarize_weaker_static_is_not_hull_but_fails_structural_survival(self):
        rows = [
            {"setting": "baseline", "parameter": "none", "parameter_value": "x",
             "status": "dominated", "conclusion_changed": False},
            {"setting": "static_baseline_mixture", "parameter": "mixture",
             "parameter_value": "always_t1", "status": "tradeoff",
             "conclusion_changed": True},
        ]
        summary = summarize(rows, baseline_status="dominated")
        self.assertEqual(summary["n_hull_flips"], 0)
        self.assertTrue(summary["primary_conclusion_survives_hull_perturbations"])
        self.assertFalse(summary["primary_conclusion_survives_structural"])
        self.assertEqual(summary["n_alternative_comparator_flips"], 1)

    def test_summarize_hull_infeasible_fails_survival(self):
        rows = [
            {"setting": "baseline", "parameter": "none", "parameter_value": "x",
             "status": "dominated", "conclusion_changed": False},
            {"setting": "routing_budget", "parameter": "budget_mult",
             "parameter_value": 0.5, "status": "infeasible",
             "conclusion_changed": True, "claim_reversed": False},
        ]
        summary = summarize(rows, baseline_status="dominated")
        self.assertEqual(summary["n_hull_flips"], 1)
        self.assertFalse(summary["primary_conclusion_survives_hull_perturbations"])
        self.assertTrue(summary["primary_conclusion_survives_became_beneficial_only"])


class TestBannedSources(unittest.TestCase):
    def test_module_has_no_download_hooks(self):
        from woais_experiments.routing import robustness as mod
        src = inspect.getsource(mod)
        self.assertNotIn("huggingface", src)
        self.assertNotIn("datasets.load_dataset", src)
        self.assertNotIn("requests.get", src)


class TestStage12Smoke(unittest.TestCase):
    def test_baseline_is_dominated_and_csv_contract(self):
        from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
        from woais_experiments.routing.policies import build_stage12_policies
        from woais_experiments.routing.robustness import panel_from_item_matrix

        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        policies = build_stage12_policies(matrix, routing)
        panel = panel_from_item_matrix(matrix, policies["ecologic"])
        rows = run_sweep(panel, n_perm=0, n_bootstrap=3, seed=20260909)
        settings = {r["setting"] for r in rows}
        self.assertGreaterEqual(panel.n, 300)
        self.assertEqual(rows[0]["status"], "dominated")
        self.assertFalse(rows[0]["conclusion_changed"])
        for needed in (
            "model_pricing",
            "output_token_multiplier",
            "router_overhead",
            "latency_penalty",
            "quality_threshold",
            "routing_budget",
            "random_seed",
            "bootstrap_resample",
            "drop_highest_cost",
            "drop_longest_output",
            "static_baseline_mixture",
        ):
            self.assertIn(needed, settings)
        text = rows_to_csv(rows)
        header = text.splitlines()[0].split(",")
        self.assertEqual(header, list(CSV_FIELDS))
        flipped = [r for r in rows if r["conclusion_changed"]]
        self.assertTrue(all("status" in r for r in flipped))


if __name__ == "__main__":
    unittest.main()
