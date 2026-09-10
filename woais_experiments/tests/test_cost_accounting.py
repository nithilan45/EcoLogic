"""Exact per-query vs naive mix×mean cost accounting."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.accounting.aggregate_cost import (
    aggregate,
    naive_mean,
    panel_from_item_matrix,
    realized_total,
    records_from_assignment,
    unconditional_model_means,
)
from woais_experiments.accounting.cost_decomposition import (
    bundle_from_assignment,
    compare_policies,
    config_cheap_expensive,
    decompose_naive_vs_realized,
    export_framework_tables,
    heterogeneous_token_panel,
    homogeneous_token_panel,
    write_tables,
)
from woais_experiments.accounting.per_query_cost import (
    UnknownPriceError,
    load_price_table,
    load_router_overhead,
    make_query,
    realized_cost,
)
from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
from woais_experiments.paths import CONFIGS, RESULTS
from woais_experiments.routing.policies import build_stage12_policies

ATOL = 1e-15
ATOL_PANEL = 1e-12


class TestConfigPrices(unittest.TestCase):
    def test_prices_come_from_models_json_not_literals_in_code(self):
        cfg = json.loads((CONFIGS / "models.json").read_text())
        prices = load_price_table(cfg)
        gpt = prices.require("gpt-4o")
        self.assertEqual(gpt.input_usd_per_million, cfg["usd_per_million"]["gpt-4o"]["input"])
        self.assertEqual(gpt.output_usd_per_million, cfg["usd_per_million"]["gpt-4o"]["output"])
        self.assertAlmostEqual(gpt.input_price_per_token, 2.50 / 1_000_000, places=18)
        self.assertAlmostEqual(gpt.output_price_per_token, 10.00 / 1_000_000, places=18)

    def test_routellm_in_out_aliases_are_loaded(self):
        prices = load_price_table()
        strong = prices.require("gpt-4-1106-preview")
        cfg = json.loads((CONFIGS / "models.json").read_text())
        self.assertEqual(
            strong.input_usd_per_million,
            cfg["routellm"]["usd_per_million"]["gpt-4-1106-preview"]["in"],
        )

    def test_unknown_model_is_refused(self):
        prices = load_price_table()
        with self.assertRaises(UnknownPriceError):
            make_query("x", "not-a-listed-model", 1, 1, prices=prices)

    def test_keyword_router_overhead_is_zero_in_config(self):
        oh = load_router_overhead("ecologic_keyword")
        self.assertEqual(oh.cost_usd, 0.0)
        self.assertEqual(oh.tokens, 0.0)


class TestPerQueryRealized(unittest.TestCase):
    def setUp(self):
        self.prices = load_price_table()
        self.oh = load_router_overhead("none")
        self.cheap, self.exp = config_cheap_expensive(self.prices)

    def test_realized_formula_matches_token_times_config_price(self):
        rec = make_query(
            "i", self.exp, 1_000, 500, prices=self.prices, overhead=self.oh, latency_ms=1140.0,
        )
        p = self.prices.require(self.exp)
        expected = 1_000 * p.input_price_per_token + 500 * p.output_price_per_token + 0.0
        self.assertEqual(rec.latency_ms, 1140.0)
        self.assertAlmostEqual(realized_cost(rec), expected, delta=ATOL)
        self.assertAlmostEqual(rec.inference_cost(), 0.0025 + 0.005, delta=ATOL)

    def test_overhead_tokens_priced_from_config_not_invented(self):
        qwen = self.prices.require("Qwen/Qwen3.5-9B")
        tokens = 250
        oh_cost = tokens * qwen.input_price_per_token
        rec = make_query(
            "i", self.cheap, 10, 20, prices=self.prices,
            overhead_cost=oh_cost, overhead_tokens=tokens,
        )
        inf = self.prices.inference_cost(self.cheap, 10, 20)
        self.assertAlmostEqual(rec.realized_cost(), inf + oh_cost, delta=ATOL)
        self.assertEqual(rec.router_overhead_tokens, tokens)

    def test_aggregate_is_exact_sum_of_individuals(self):
        recs = [
            make_query(f"q{i}", self.cheap, 10 * (i + 1), 5 * (i + 1), prices=self.prices)
            for i in range(7)
        ]
        total = realized_total(recs)
        self.assertAlmostEqual(total, math.fsum(r.realized_cost() for r in recs), delta=ATOL)
        self.assertAlmostEqual(
            aggregate(recs)["realized_total"], total, delta=ATOL,
        )


class TestNaiveEqualsRealized(unittest.TestCase):
    def setUp(self):
        self.prices = load_price_table()
        self.oh = load_router_overhead("none")
        self.cheap, self.exp = config_cheap_expensive(self.prices)

    def test_zero_within_model_variance(self):
        panel = homogeneous_token_panel(self.prices)
        assign = {
            q: (self.cheap if i % 2 == 0 else self.exp)
            for i, q in enumerate(panel.query_ids)
        }
        recs, decomp = bundle_from_assignment(panel, assign, self.prices, self.oh, atol=ATOL)
        self.assertTrue(decomp["reconciles"])
        self.assertAlmostEqual(
            decomp["naive_inference_mean"], decomp["realized_inference_mean"], delta=ATOL,
        )
        self.assertAlmostEqual(decomp["covariance_sum"], 0.0, delta=ATOL)
        self.assertFalse(decomp["heterogeneity_needed"])
        self.assertEqual(len(recs), 8)

    def test_static_policy_on_heterogeneous_costs(self):
        panel = heterogeneous_token_panel(self.prices)
        assign = {q: self.cheap for q in panel.query_ids}
        recs, decomp = bundle_from_assignment(panel, assign, self.prices, self.oh, atol=ATOL)
        means = unconditional_model_means(panel, self.prices)
        self.assertTrue(decomp["reconciles"])
        self.assertAlmostEqual(
            decomp["realized_inference_mean"], means[self.cheap], delta=ATOL,
        )
        self.assertAlmostEqual(
            decomp["naive_inference_mean"], means[self.cheap], delta=ATOL,
        )
        self.assertAlmostEqual(realized_total(recs), means[self.cheap] * 6, delta=ATOL_PANEL)


class TestNaiveBiasDirection(unittest.TestCase):
    def setUp(self):
        self.prices = load_price_table()
        self.oh = load_router_overhead("none")
        self.cheap, self.exp = config_cheap_expensive(self.prices)
        self.panel = heterogeneous_token_panel(self.prices)
        self.under_assign = {
            "q0": self.cheap, "q1": self.cheap, "q2": self.exp,
            "q3": self.cheap, "q4": self.cheap, "q5": self.cheap,
        }
        self.over_assign = {
            "q0": self.cheap, "q1": self.cheap, "q2": self.cheap,
            "q3": self.exp, "q4": self.exp, "q5": self.exp,
        }

    def test_naive_underestimates_when_long_items_stay_on_cheap_model(self):
        _, decomp = bundle_from_assignment(
            self.panel, self.under_assign, self.prices, self.oh, atol=ATOL,
        )
        self.assertTrue(decomp["reconciles"])
        self.assertTrue(decomp["naive_underestimates_realized"])
        self.assertGreater(decomp["covariance_sum"], 0.0)
        self.assertAlmostEqual(
            decomp["realized_minus_naive"], decomp["covariance_sum"], delta=ATOL,
        )

    def test_naive_overestimates_when_only_short_items_go_to_cheap_model(self):
        _, decomp = bundle_from_assignment(
            self.panel, self.over_assign, self.prices, self.oh, atol=ATOL,
        )
        self.assertTrue(decomp["reconciles"])
        self.assertTrue(decomp["naive_overestimates_realized"])
        self.assertLess(decomp["covariance_sum"], 0.0)

    def test_identity_holds_to_float_tolerance_on_both_policies(self):
        for assign in (self.under_assign, self.over_assign):
            recs, decomp = bundle_from_assignment(
                self.panel, assign, self.prices, self.oh, atol=ATOL,
            )
            naive = naive_mean(decomp["mix_fractions"], decomp["unconditional_model_means"])
            self.assertAlmostEqual(naive, decomp["naive_inference_mean"], delta=ATOL)
            rhs = naive + decomp["covariance_sum"]
            self.assertAlmostEqual(decomp["realized_inference_mean"], rhs, delta=ATOL)
            self.assertAlmostEqual(
                decomp["abs_residual"],
                abs(decomp["realized_inference_mean"] - rhs),
                delta=ATOL,
            )
            self.assertEqual(len(recs), 6)


class TestSignFlip(unittest.TestCase):
    def test_naive_prefers_length_biased_router_realized_does_not(self):
        prices = load_price_table()
        oh = load_router_overhead("none")
        cheap, exp = config_cheap_expensive(prices)
        panel = heterogeneous_token_panel(prices)
        a = {
            "q0": cheap, "q1": cheap, "q2": exp,
            "q3": cheap, "q4": cheap, "q5": cheap,
        }
        b = {
            "q0": cheap, "q1": cheap, "q2": cheap,
            "q3": exp, "q4": exp, "q5": exp,
        }
        _, da = bundle_from_assignment(panel, a, prices, oh, atol=ATOL)
        _, db = bundle_from_assignment(panel, b, prices, oh, atol=ATOL)
        cmp = compare_policies("length_biased", da, "short_to_cheap", db, atol=ATOL)
        self.assertTrue(cmp["sign_flip"])
        self.assertEqual(cmp["naive_prefers"], "length_biased")
        self.assertEqual(cmp["realized_prefers"], "short_to_cheap")
        self.assertLess(da["naive_inference_mean"], db["naive_inference_mean"])
        self.assertGreater(da["realized_inference_mean"], db["realized_inference_mean"])


class TestTablesAndStage12(unittest.TestCase):
    def test_write_tables_csv_json_roundtrip(self):
        prices = load_price_table()
        oh = load_router_overhead("none")
        panel = homogeneous_token_panel(prices)
        cheap, exp = config_cheap_expensive(prices)
        assign = {q: cheap if i < 4 else exp for i, q in enumerate(panel.query_ids)}
        recs, decomp = bundle_from_assignment(panel, assign, prices, oh, atol=ATOL)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            paths = write_tables(
                label="roundtrip", records=recs, dest_dir=dest, panel=panel, prices=prices,
            )
            rows = (paths["per_query_csv"]).read_text().strip().splitlines()
            self.assertEqual(len(rows), 1 + len(recs))
            self.assertIn("realized_cost", rows[0])
            blob = json.loads(paths["aggregate_json"].read_text())
            self.assertEqual(blob["n"], 8)
            self.assertAlmostEqual(
                blob["realized_total"], realized_total(recs), delta=ATOL,
            )
            dec = json.loads(paths["decomposition_json"].read_text())
            self.assertTrue(dec["reconciles"])
            self.assertAlmostEqual(
                dec["covariance_sum"], decomp["covariance_sum"], delta=ATOL,
            )

    def test_export_framework_tables_and_stage12_identity(self):
        prices = load_price_table()
        oh = load_router_overhead("ecologic_keyword")
        self.assertEqual(oh.cost_usd, 0.0)
        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        policies = build_stage12_policies(matrix, routing)
        panel, eco = panel_from_item_matrix(matrix, policies["ecologic"])
        _, t2 = panel_from_item_matrix(matrix, policies["always_t2"])
        dest = RESULTS / "accounting" / "framework"
        written = export_framework_tables(
            dest, prices=prices, overhead=oh, stage12=(panel, eco, t2),
        )
        self.assertIn("naive_underestimates", written)
        self.assertTrue((dest / "sign_flip_comparison.json").exists())

        recs = records_from_assignment(panel, eco, prices, oh)
        decomp = decompose_naive_vs_realized(recs, panel, prices, atol=ATOL_PANEL)
        self.assertTrue(decomp["reconciles"], msg=decomp["abs_residual"])
        self.assertEqual(decomp["n"], 364)
        self.assertTrue(decomp["naive_underestimates_realized"])

        recs_t2 = records_from_assignment(panel, t2, prices, oh)
        decomp_t2 = decompose_naive_vs_realized(recs_t2, panel, prices, atol=ATOL_PANEL)
        self.assertTrue(decomp_t2["reconciles"])
        self.assertAlmostEqual(
            decomp_t2["naive_inference_mean"],
            decomp_t2["realized_inference_mean"],
            delta=ATOL_PANEL,
        )

        # Recomputed inference USD matches the stored per-call `usd` field.
        model = matrix.model_of[2]
        qid = matrix.item_ids[0]
        stored = matrix.usd[(2, qid)]
        recomputed = prices.inference_cost(
            model, matrix.prompt_tokens[(2, qid)], matrix.completion_tokens[(2, qid)],
        )
        self.assertAlmostEqual(stored, recomputed, delta=1e-12)

    def test_constant_overhead_does_not_change_naive_realized_gap(self):
        prices = load_price_table()
        panel = heterogeneous_token_panel(prices)
        cheap, _ = config_cheap_expensive(prices)
        qwen = prices.require("Qwen/Qwen3.5-9B")
        oh_cost = 40 * qwen.input_price_per_token
        assign = {q: cheap for q in panel.query_ids}
        recs0 = records_from_assignment(panel, assign, prices)
        recs1 = [
            make_query(
                r.query_id, r.selected_model, r.input_tokens, r.output_tokens,
                prices=prices, overhead_cost=oh_cost, overhead_tokens=40,
                latency_ms=r.latency_ms,
            )
            for r in recs0
        ]
        d0 = decompose_naive_vs_realized(recs0, panel, prices, atol=ATOL)
        d1 = decompose_naive_vs_realized(recs1, panel, prices, atol=ATOL)
        self.assertAlmostEqual(d0["realized_minus_naive"], d1["realized_minus_naive"], delta=ATOL)
        self.assertAlmostEqual(
            aggregate(recs1)["realized_mean"] - aggregate(recs0)["realized_mean"],
            oh_cost,
            delta=ATOL,
        )


if __name__ == "__main__":
    unittest.main()
