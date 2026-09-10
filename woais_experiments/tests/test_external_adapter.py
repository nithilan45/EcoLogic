"""Generic external routing-benchmark adapter: schema, ingest, audit."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.external.adapter import (
    ROUTELLM_STRONG,
    ROUTELLM_WEAK,
    discover_sources,
    from_routellm_responses,
    load_source,
    normalize_records,
    to_wide,
)
from woais_experiments.external.run_external_audit import (
    audit_panel,
    naive_vs_realized,
    pick_oracle_method,
    run_records,
)
from woais_experiments.external.schema import MAX_MODELS, MIN_MODELS, SchemaError, LongPanel
from woais_experiments.external import run_external_audit as audit_mod


def _two_model_records():
    # A=(1, 0.5), B=(3, 0.9) means; oracle [A, B] on two queries.
    return [
        {"query_id": "q1", "model": "A", "correctness": 1, "cost": 1.0, "latency_ms": 5, "input_tokens": 10, "output_tokens": 2, "router_assignment": "A"},
        {"query_id": "q1", "model": "B", "correctness": 1, "cost": 3.0, "latency_ms": 40, "input_tokens": 10, "output_tokens": 20, "router_assignment": "A"},
        {"query_id": "q2", "model": "A", "correctness": 0, "cost": 1.0, "latency_s": 0.015, "prompt_tokens": 10, "completion_tokens": 4, "router_assignment": "B"},
        {"query_id": "q2", "model": "B", "correctness": 1, "cost": 3.0, "latency_s": 0.04, "prompt_tokens": 10, "completion_tokens": 20, "router_assignment": "B"},
    ]


class TestSchemaAndAliases(unittest.TestCase):
    def test_aliases_and_latency_units(self):
        panel = normalize_records(_two_model_records(), dataset="toy")
        self.assertEqual(panel.n_models, 2)
        self.assertEqual(panel.n_queries, 2)
        by = {(r.query_id, r.model): r for r in panel.rows}
        self.assertAlmostEqual(by[("q2", "A")].latency_ms, 15.0)
        self.assertEqual(by[("q2", "A")].input_tokens, 10)
        self.assertEqual(panel.assignment_map(), {"q1": "A", "q2": "B"})

    def test_rejects_one_model(self):
        recs = [
            {"query_id": "q", "model": "only", "quality": 1, "cost": 1},
            {"query_id": "r", "model": "only", "quality": 0, "cost": 1},
        ]
        with self.assertRaises(SchemaError):
            normalize_records(recs, dataset="one")

    def test_rejects_more_than_fifty_models(self):
        recs = []
        for j in range(MAX_MODELS + 1):
            recs.append({"query_id": "q", "model": f"m{j:02d}", "quality": 0.5, "cost": 1.0})
        with self.assertRaises(SchemaError):
            normalize_records(recs, dataset="too_many")


class TestAuditTwoToFifty(unittest.TestCase):
    def test_two_model_full_stack(self):
        out = run_records(_two_model_records(), dataset="toy", n_boot=200, seed=20260909, n_grid=5)
        self.assertTrue(out["usd_available"])
        self.assertEqual(out["n_models"], 2)
        self.assertTrue(out["accounting"]["naive_vs_realized"]["reconciles"])
        self.assertIn("quality_advantage", out["cost_matching"])
        self.assertAlmostEqual(out["cost_matching"]["cost_savings"], 1.0)
        self.assertTrue(out["oracle_frontier"]["unconstrained_oracle"]["feasible"])
        self.assertTrue(out["bootstrap"]["available"])
        self.assertGreater(out["bootstrap"]["quality_advantage"]["point"], 0)
        self.assertIsNone(out["bootstrap"]["significant_05_quality"])
        self.assertTrue(out["bootstrap"]["ci_is_not_a_hypothesis_test"])

    def test_naive_identity_on_arrays(self):
        cost = np.array([[1.0, 3.0], [1.0, 3.0]])
        choice = np.array([0, 1])
        rec = naive_vs_realized(cost, choice, ["A", "B"])
        self.assertTrue(rec["reconciles"])
        # mix 50/50, means 1 and 3 → naive 2; realized (1+3)/2 = 2
        self.assertAlmostEqual(rec["naive_inference_mean"], 2.0)
        self.assertAlmostEqual(rec["realized_inference_mean"], 2.0)

    def test_eight_models(self):
        recs = []
        models = [f"m{j}" for j in range(8)]
        for i in range(12):
            for j, m in enumerate(models):
                recs.append({
                    "item_id": f"q{i}",
                    "model": m,
                    "quality_score": 0.1 * j + 0.01 * i,
                    "usd": 0.5 + 0.2 * j + 0.01 * (i % 3),
                    "router_assignment": models[i % 8],
                })
        panel = normalize_records(recs, dataset="m8")
        self.assertEqual(panel.n_models, 8)
        out = audit_panel(panel, n_boot=0, n_grid=4, oracle_method="approx")
        self.assertEqual(out["n_models"], 8)
        self.assertTrue(out["usd_available"])
        self.assertIn("always_cheapest", out["static_baselines"]["catalog"])
        self.assertEqual(out["oracle_frontier"]["n_models"], 8)

    def test_fifty_models_smoke(self):
        recs = []
        models = [f"m{j:02d}" for j in range(50)]
        self.assertEqual(len(models), MAX_MODELS)
        rng = np.random.default_rng(20260909)
        n = 6
        for i in range(n):
            pick = int(rng.integers(0, 50))
            for j, m in enumerate(models):
                recs.append({
                    "query_id": f"q{i}",
                    "model": m,
                    "quality": float(rng.random()),
                    "cost": 0.1 + 0.05 * j + 0.01 * float(rng.random()),
                    "router_assignment": models[pick],
                })
        panel = normalize_records(recs, dataset="m50")
        self.assertEqual(panel.n_models, 50)
        wide = to_wide(panel)
        self.assertEqual(wide.m, 50)
        self.assertEqual(wide.n, n)
        self.assertEqual(pick_oracle_method(wide.n, wide.m), "approx")
        out = audit_panel(panel, n_boot=0, n_grid=3, oracle_method="approx")
        self.assertEqual(out["n_models"], 50)
        self.assertTrue(out["usd_available"])
        self.assertEqual(out["oracle_frontier"]["n_models"], 50)
        self.assertIn("omitted", out["static_baselines"]["catalog"]["pairwise_random"])

    def test_quality_only_when_cost_missing(self):
        recs = [
            {"query_id": "q1", "model": "A", "correctness": 1},
            {"query_id": "q1", "model": "B", "correctness": 0},
            {"query_id": "q2", "model": "A", "correctness": 0},
            {"query_id": "q2", "model": "B", "correctness": 1},
        ]
        out = run_records(recs, dataset="qonly", n_boot=0)
        self.assertFalse(out["usd_available"])
        self.assertIn("naive accounting", " ".join(out["skipped"]))


class TestRouteLLMAndDiscovery(unittest.TestCase):
    def test_melt_wide_gsm8k_shape(self):
        rows = [
            {ROUTELLM_WEAK: True, ROUTELLM_STRONG: True, "prompt": "1+1?"},
            {ROUTELLM_WEAK: False, ROUTELLM_STRONG: True, "prompt": "hard"},
        ]
        panel = from_routellm_responses(rows, dataset="rl")
        self.assertEqual(panel.n_models, 2)
        self.assertEqual(panel.n_queries, 2)
        self.assertEqual(set(panel.models), {ROUTELLM_WEAK, ROUTELLM_STRONG})
        out = audit_panel(panel, n_boot=0)
        self.assertFalse(out["usd_available"])

    def test_discover_does_not_require_clone(self):
        found = {s.name: s for s in discover_sources()}
        self.assertTrue(found["routellm_s9_committed"].available)
        self.assertFalse(found["routellm_gsm8k_responses"].available)
        self.assertIn("Not downloaded", found["routellm_gsm8k_responses"].note)
        blob = load_source("routellm_s9_committed")
        self.assertFalse(blob["panel_available"])
        self.assertEqual(blob["n_models"], 2)
        self.assertEqual(blob["summary"]["n_items"], 1307)

    def test_stage12_through_adapter(self):
        panel = load_source("ecologic_stage12")
        self.assertIsInstance(panel, LongPanel)
        self.assertGreaterEqual(panel.n_models, MIN_MODELS)
        self.assertLessEqual(panel.n_models, MAX_MODELS)
        self.assertEqual(panel.n_queries, 364)
        out = audit_panel(panel, n_boot=0, n_grid=5, oracle_method="approx")
        self.assertTrue(out["usd_available"])
        self.assertTrue(out["accounting"]["naive_vs_realized"]["reconciles"])
        cm = out["cost_matching"]
        self.assertIsNotNone(cm)
        self.assertTrue(cm["same_quality_feasible"])
        self.assertLess(cm["cost_savings"], 0.0)

    def test_csv_roundtrip(self):
        panel = normalize_records(_two_model_records(), dataset="toy")
        text = panel.to_csv()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            path.write_text(text)
            from woais_experiments.external.adapter import from_path
            again = from_path(path, dataset="toy")
        self.assertEqual(again.n_models, 2)
        self.assertEqual(again.n_queries, 2)

    def test_audit_module_does_not_download_or_fit(self):
        src = inspect.getsource(audit_mod)
        for banned in ("httpx", "chat(", "huggingface", "tiktoken", "transformers", "sklearn", "torch"):
            self.assertNotIn(banned, src)
        self.assertNotIn("urlopen", src)
        self.assertNotIn("requests.", src)

    def test_bare_latency_column_requires_explicit_unit(self):
        recs = [
            {"query_id": "q1", "model": "A", "quality": 1, "cost": 1.0, "latency": 5},
            {"query_id": "q1", "model": "B", "quality": 1, "cost": 3.0, "latency": 10},
            {"query_id": "q2", "model": "A", "quality": 0, "cost": 1.0, "latency": 5},
            {"query_id": "q2", "model": "B", "quality": 1, "cost": 3.0, "latency": 10},
        ]
        with self.assertRaises(SchemaError):
            normalize_records(recs, dataset="amb")
        panel = normalize_records(recs, dataset="amb", latency_unit="ms")
        by = {(r.query_id, r.model): r for r in panel.rows}
        self.assertAlmostEqual(by[("q1", "A")].latency_ms, 5.0)

    def test_price_table_fill_is_not_labeled_metered(self):
        recs = [
            {"query_id": "q1", "model": "Qwen/Qwen3.5-9B", "quality": 1, "input_tokens": 100, "output_tokens": 10},
            {"query_id": "q1", "model": "gpt-4o", "quality": 1, "input_tokens": 100, "output_tokens": 10},
            {"query_id": "q2", "model": "Qwen/Qwen3.5-9B", "quality": 0, "input_tokens": 100, "output_tokens": 10},
            {"query_id": "q2", "model": "gpt-4o", "quality": 1, "input_tokens": 100, "output_tokens": 10},
        ]
        panel = normalize_records(recs, dataset="fill")
        self.assertTrue(all(r.cost_source == "price_table" for r in panel.rows))
        self.assertTrue(all(r.realized_cost is not None for r in panel.rows))


if __name__ == "__main__":
    unittest.main()
