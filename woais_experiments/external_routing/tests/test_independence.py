"""Assignments must be independent of test-set correctness labels."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import shutil
import unittest

import numpy as np

from woais_experiments.external_routing.audit_router import (
    classify_threshold,
    evaluate_threshold,
    naive_realized,
)
from woais_experiments.external_routing.panel import from_arrays
from woais_experiments.external_routing.reconstruct_assignments import (
    EXTERNAL_LEARNED_ROUTER,
    ORACLE_ASSIGNMENT,
    oracle_route,
    route_by_score,
)
from woais_experiments.external_routing.routellm_adapter import inventory
from woais_experiments.external_routing.run_threshold_sweep import (
    GRID_PATH,
    QUERY_CSV,
    THRESHOLD_CSV,
    load_grid,
    run_sweep,
    write_incomplete_artifacts,
)
from woais_experiments.paths import RESULTS, reset_run_context, set_run_context


def _score_from_id(qid: str) -> float:
    h = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _toy_panel(*, n: int = 40, seed: int = 0):
    ids = [f"q{i:03d}" for i in range(n)]
    scores = np.array([_score_from_id(q) for q in ids], dtype=float)
    rng = np.random.default_rng(seed)
    # Labels are a different random stream than scores.
    q0 = rng.integers(0, 2, size=n).astype(float)
    q1 = np.clip(q0 + rng.integers(0, 2, size=n), 0, 1).astype(float)
    i0 = np.full(n, 20.0)
    o0 = np.full(n, 10.0)
    i1 = np.full(n, 20.0)
    o1 = 10.0 + 40.0 * scores
    c0 = 0.001 * np.ones(n)
    c1 = 0.002 + 0.008 * scores
    return from_arrays(
        router_name="toy_external",
        dataset="toy",
        query_ids=ids,
        scores=scores,
        cheap_model="cheap_lm",
        strong_model="strong_lm",
        cheap_quality=q0,
        strong_quality=q1,
        cheap_input_tokens=i0,
        cheap_output_tokens=o0,
        strong_input_tokens=i1,
        strong_output_tokens=o1,
        cheap_cost=c0,
        strong_cost=c1,
        score_provenance="sha256(query_id) — independent of labels",
        source="unit_test",
    )


class TestAssignmentIndependence(unittest.TestCase):
    def test_route_by_score_signature_has_no_labels(self):
        names = inspect.signature(route_by_score).parameters
        for banned in ("quality", "correct", "correctness", "label", "cost", "y"):
            self.assertNotIn(banned, names)
        self.assertNotIn("quality", route_by_score.__code__.co_varnames)

    def test_shuffling_labels_does_not_change_assignment(self):
        panel = _toy_panel()
        a = route_by_score(panel.scores, 0.4, cheap=panel.cheap_model, strong=panel.strong_model)
        shuffled_q0 = panel.cheap_quality.copy()
        np.random.default_rng(1).shuffle(shuffled_q0)
        b = route_by_score(panel.scores, 0.4, cheap=panel.cheap_model, strong=panel.strong_model)
        self.assertTrue(np.array_equal(a, b))
        panel.cheap_quality[:] = shuffled_q0
        panel.strong_quality[:] = 1.0 - panel.strong_quality
        c = route_by_score(panel.scores, 0.4, cheap=panel.cheap_model, strong=panel.strong_model)
        self.assertTrue(np.array_equal(a, c))

    def test_scores_are_functions_of_ids_not_correctness(self):
        panel = _toy_panel(seed=0)
        other = _toy_panel(seed=99)
        self.assertTrue(np.allclose(panel.scores, other.scores))
        self.assertFalse(np.array_equal(panel.cheap_quality, other.cheap_quality))

    def test_oracle_changes_with_labels_learned_route_does_not(self):
        panel = _toy_panel()
        learned = route_by_score(panel.scores, 0.5, cheap=panel.cheap_model, strong=panel.strong_model)
        hindsight = oracle_route(
            panel.cheap_quality, panel.strong_quality, panel.cheap_cost, panel.strong_cost,
            cheap=panel.cheap_model, strong=panel.strong_model,
        )
        flipped = oracle_route(
            1.0 - panel.cheap_quality,
            1.0 - panel.strong_quality,
            panel.cheap_cost,
            panel.strong_cost,
            cheap=panel.cheap_model,
            strong=panel.strong_model,
        )
        rec = evaluate_threshold(panel, 0.5, n_boot=40, n_perm=39, seed=1)
        self.assertEqual(rec["assignment_kind"], EXTERNAL_LEARNED_ROUTER)
        self.assertEqual(rec["policies"]["oracle_assignment"]["kind"], ORACLE_ASSIGNMENT)
        self.assertEqual(rec["policies"]["external_learned_router"]["kind"], EXTERNAL_LEARNED_ROUTER)
        self.assertFalse(np.array_equal(learned, hindsight))
        self.assertFalse(np.array_equal(hindsight, flipped))
        self.assertTrue(np.array_equal(
            learned,
            route_by_score(panel.scores, 0.5, cheap=panel.cheap_model, strong=panel.strong_model),
        ))


class TestAccountingAndGrid(unittest.TestCase):
    def test_naive_equals_realized_for_constant_costs(self):
        n = 20
        side = np.array([0] * 10 + [1] * 10)
        c0 = np.full(n, 2.0)
        c1 = np.full(n, 5.0)
        out = naive_realized(side, c0, c1)
        self.assertAlmostEqual(out["naive_cost"], out["realized_cost"])
        self.assertAlmostEqual(out["absolute_accounting_error"], 0.0)

    def test_predefined_grid_is_frozen(self):
        grid = load_grid()
        self.assertTrue(grid["frozen_before_quality_cost"])
        self.assertEqual(len(grid["thresholds"]), 21)
        self.assertEqual(grid["thresholds"][0], 0.0)
        self.assertEqual(grid["thresholds"][-1], 1.0)
        raw = json.loads(GRID_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw["thresholds"], grid["thresholds"])
        self.assertNotIn("qcut", str(grid.get("kind", "")).lower())

    def test_sweep_exports_every_threshold(self):
        panel = _toy_panel(n=24)
        grid = {
            "frozen_before_quality_cost": True,
            "thresholds": [0.0, 0.5, 1.0],
            "kind": "test_grid",
        }
        summary = run_sweep(panel, grid, n_boot=32, n_perm=31, seed=2, write=False)
        self.assertEqual(len(summary["rows"]), 3)
        self.assertEqual([r["threshold"] for r in summary["rows"]], [0.0, 0.5, 1.0])
        self.assertIn("claim_beats_cost_matched_static", summary)
        flags = [classify_threshold(r) for r in summary["rows"]]
        self.assertEqual(len(flags), 3)
        self.assertEqual(summary["router_kind"], EXTERNAL_LEARNED_ROUTER)
        self.assertEqual(summary["oracle_kind"], ORACLE_ASSIGNMENT)

    def test_incomplete_artifacts_do_not_invent_assignments(self):
        root = RESULTS / "_external_routing_unittest"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        tokens = set_run_context(output_root=root, overwrite_policy="replace")
        try:
            payload = write_incomplete_artifacts(
                "unit-test missing scores",
                gsm8k_rows=[
                    {
                        "query_id": "gsm8k-0000",
                        "cheap_correct": 1.0,
                        "strong_correct": 0.0,
                    }
                ],
            )
            qpath = root / QUERY_CSV
            tpath = root / THRESHOLD_CSV
            with open(qpath, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cheap_quality"], "1.0")
            self.assertEqual(str(rows[0]["router_assignment"]).strip(), "")
            self.assertEqual(str(rows[0]["router_score"]).strip(), "")
            self.assertEqual(str(rows[0]["selected_model"]).strip(), "")
            with open(tpath, encoding="utf-8", newline="") as fh:
                trows = list(csv.DictReader(fh))
            self.assertEqual(trows, [])
            self.assertFalse(payload["query_level_assignments_available"])
            self.assertNotIn("sweep", payload)
        finally:
            reset_run_context(tokens)
            shutil.rmtree(root, ignore_errors=True)

    def test_inventory_does_not_invent_query_level_files(self):
        inv = inventory()
        self.assertFalse(inv["committed_aggregates"]["s9_static_baselines"]["query_level"])
        self.assertFalse(inv["committed_aggregates"]["s9_static_baselines"]["has_router_scores"])
        self.assertFalse(inv["committed_aggregates"]["s9_static_baselines"]["has_assignments"])
        self.assertIn("published numeric token counts", inv["not_available_anywhere"])
        self.assertIn("transformers", inv["scoring_dependencies"])
        self.assertTrue(inv["historical_stage9_aggregates"]["not_this_protocol"])

    def test_naive_wins_exact_loses_and_ranking_flip(self):
        policies = {
            "always_cheap": {
                "kind": "baseline", "quality": 0.4, "realized_cost": 1.0, "naive_cost": 1.0,
            },
            "always_strong": {
                "kind": "baseline", "quality": 0.9, "realized_cost": 10.0, "naive_cost": 10.0,
            },
            "random_mixture": {
                "kind": "baseline", "quality": 0.6, "realized_cost": 5.0, "naive_cost": 5.0,
            },
            "cost_matched_static": {
                "kind": "baseline", "quality": 0.61, "realized_cost": 4.5, "naive_cost": 4.5,
            },
            "external_learned_router": {
                "kind": EXTERNAL_LEARNED_ROUTER,
                "quality": 0.6,
                "realized_cost": 6.0,
                "naive_cost": 4.0,
            },
            "oracle_assignment": {
                "kind": ORACLE_ASSIGNMENT, "quality": 0.95, "realized_cost": 2.0, "naive_cost": 2.0,
            },
        }
        row = {
            "threshold": 0.5,
            "naive_cost": 4.0,
            "realized_cost": 6.0,
            "absolute_accounting_error": -2.0,
            "relative_accounting_error": -2.0 / 6.0,
            "quality_advantage_vs_cost_matched_static_naive_cost": 0.05,
            "quality_advantage_vs_cost_matched_static": -0.01,
            "cost_regret_naive_vs_oracle": -0.1,
            "cost_regret_realized_vs_oracle": 0.2,
            "policies": policies,
        }
        flags = classify_threshold(row)
        self.assertTrue(flags["naive_wins_exact_loses"])
        self.assertTrue(flags["accounting_sign_flip"])
        self.assertTrue(flags["ranking_flip"])
        self.assertFalse(flags["negligible_accounting_difference"])


class TestGenericSecondRouter(unittest.TestCase):
    def test_from_arrays_accepts_another_router(self):
        panel = from_arrays(
            router_name="other_router",
            dataset="other",
            query_ids=["a", "b"],
            scores=[0.1, 0.9],
            cheap_model="m0",
            strong_model="m1",
            cheap_quality=[1, 0],
            strong_quality=[1, 1],
            cheap_input_tokens=[1, 1],
            cheap_output_tokens=[1, 1],
            strong_input_tokens=[1, 1],
            strong_output_tokens=[2, 2],
            cheap_cost=[0.1, 0.1],
            strong_cost=[1.0, 1.0],
        )
        assign = route_by_score(panel.scores, 0.5, cheap="m0", strong="m1")
        self.assertEqual(list(assign), ["m0", "m1"])
        self.assertEqual(panel.router_name, "other_router")
