"""xRouteBench adapter: mapping, missing fields, resume. No live Hub download."""

from __future__ import annotations

import inspect
import shutil
import unittest
from pathlib import Path

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.external.run_xroutebench import (
    Job,
    combo_is_complete,
    relative_accounting_error,
    run_audit,
    run_combo,
)
from woais_experiments.external.xroutebench import (
    CandidatePrices,
    classify_config,
    detect_query_id_scheme,
    discover_catalog,
    map_routing_records,
    routing_panel_from_records,
)
from woais_experiments.frozen import write_result
from woais_experiments.paths import RESULTS


def _prices():
    return {
        "cheap": CandidatePrices("cheap", input_price_per_1m=1.0, output_price_per_1m=2.0),
        "strong": CandidatePrices("strong", input_price_per_1m=10.0, output_price_per_1m=20.0),
    }


def _row(
    task_name,
    task_id,
    model,
    performance,
    *,
    input_tokens=10,
    output_tokens=20,
    response_time=0.05,
    embedding_id="emb-ignore",
    **extra,
):
    rec = {
        "task_name": task_name,
        "task_id": task_id,
        "model_name": model,
        "performance": performance,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_time": response_time,
        "embedding_id": embedding_id,
        "token_num": 9999,
        "metric": "accuracy",
        "id": "generic-id-must-not-win",
    }
    rec.update(extra)
    return rec


def _balanced_panel(n_queries=2):
    rows = []
    for i in range(n_queries):
        # Length-biased: oracle picks cheap on even, strong on odd, with unequal costs.
        if i % 2 == 0:
            rows.append(_row("gsm8k", i, "cheap", 1.0, input_tokens=10, output_tokens=2, response_time=0.01))
            rows.append(_row("gsm8k", i, "strong", 0.4, input_tokens=10, output_tokens=50, response_time=0.4))
        else:
            rows.append(_row("gsm8k", i, "cheap", 0.0, input_tokens=10, output_tokens=40, response_time=0.2))
            rows.append(_row("gsm8k", i, "strong", 1.0, input_tokens=10, output_tokens=8, response_time=0.08))
    return rows


class TestClassifyAndIds(unittest.TestCase):
    def test_classify_from_features_not_a_fixed_list(self):
        self.assertEqual(
            classify_config(
                "unexpected_panel",
                ["task_id", "model_name", "performance", "input_tokens"],
            ),
            "routing",
        )
        self.assertEqual(
            classify_config("llm_candidates", ["model_name", "input_price_per_1m", "output_price_per_1m"]),
            "pricing",
        )
        self.assertEqual(classify_config("memory_locomo_queries", ["query", "task_id"]), "queries_only")
        self.assertEqual(classify_config("brand_new_queries", []), "queries_only")
        self.assertEqual(
            classify_config("personalized", ["model_1", "model_2", "judge", "query", "task_id"]),
            "pairwise",
        )

    def test_discover_catalog_uses_injected_names(self):
        def get_names(repo):
            self.assertEqual(repo, "org/demo")
            return ["llm_candidates", "llmrouter_generic", "foo_queries", "video"]

        def inspect(repo, name, cache):
            feats = {
                "llm_candidates": ["model_name", "input_price_per_1m", "output_price_per_1m"],
                "llmrouter_generic": ["task_id", "task_name", "model_name", "performance"],
                "foo_queries": ["query", "task_id"],
                "video": ["task_id", "model_name", "performance"],
            }[name]
            splits = ["train"] if name == "llm_candidates" else ["train", "test"]
            return feats, splits

        cat = discover_catalog("org/demo", get_names=get_names, inspect_config=inspect)
        kinds = {c["name"]: c["kind"] for c in cat}
        self.assertEqual(kinds["llm_candidates"], "pricing")
        self.assertEqual(kinds["llmrouter_generic"], "routing")
        self.assertEqual(kinds["foo_queries"], "queries_only")
        self.assertEqual(kinds["video"], "routing")
        self.assertEqual(cat[1]["splits"], ["train", "test"])

    def test_preserves_task_id_when_unique(self):
        recs = _balanced_panel(2)
        self.assertEqual(detect_query_id_scheme(recs), "task_id")
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        self.assertEqual(meta["query_id_scheme"], "task_id")
        self.assertEqual({r.query_id for r in rows}, {"0", "1"})
        self.assertTrue(all(r.query_id != "generic-id-must-not-win" for r in rows))
        self.assertTrue(all(r.query_id != "emb-ignore" for r in rows))

    def test_concatenates_original_fields_on_task_id_collision(self):
        recs = [
            _row("gsm8k", 0, "cheap", 1.0),
            _row("gsm8k", 0, "strong", 0.0),
            _row("humaneval", 0, "cheap", 0.0),
            _row("humaneval", 0, "strong", 1.0),
        ]
        self.assertEqual(detect_query_id_scheme(recs), "task_name::task_id")
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        self.assertEqual(meta["query_id_scheme"], "task_name::task_id")
        self.assertEqual({r.query_id for r in rows}, {"gsm8k::0", "humaneval::0"})

    def test_missing_task_id_uses_original_query_not_embedding_id(self):
        recs = [
            {
                "task_name": "math",
                "task_id": None,
                "query": "What is 1+1?",
                "embedding_id": 99,
                "model_name": "cheap",
                "performance": 1.0,
                "input_tokens": 10,
                "output_tokens": 2,
            },
            {
                "task_name": "math",
                "task_id": None,
                "query": "What is 1+1?",
                "embedding_id": 99,
                "model_name": "strong",
                "performance": 0.0,
                "input_tokens": 10,
                "output_tokens": 2,
            },
        ]
        self.assertEqual(detect_query_id_scheme(recs), "task_name::task_id_or_query")
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        self.assertEqual(meta["query_id_scheme"], "task_name::task_id_or_query")
        self.assertEqual({r.query_id for r in rows}, {"math::What is 1+1?"})
        self.assertTrue(all(r.query_id != "99" for r in rows))

    def test_colliding_task_id_appends_original_embedding_id(self):
        recs = [
            _row("aime", "AIME-2020-4", "cheap", 0.0, embedding_id=175, input_tokens=153, output_tokens=6),
            _row("aime", "AIME-2020-4", "strong", 0.0, embedding_id=175, input_tokens=153, output_tokens=6),
            _row("aime", "AIME-2020-4", "cheap", 1.0, embedding_id=205, input_tokens=164, output_tokens=6),
            _row("aime", "AIME-2020-4", "strong", 0.0, embedding_id=205, input_tokens=164, output_tokens=6),
        ]
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        self.assertIn("disambiguate", meta["query_id_scheme"])
        ids = {r.query_id for r in rows}
        self.assertEqual(len(ids), 2)
        self.assertTrue(any("AIME-2020-4" in q and "embedding_id=175" in q for q in ids))
        self.assertTrue(any("AIME-2020-4" in q and "embedding_id=205" in q for q in ids))
        panel, info = routing_panel_from_records(recs, dataset="x", prices=_prices(), min_queries=2)
        self.assertIsNotNone(panel)
        self.assertEqual(panel.n_queries, 2)


class TestMissingFieldsAndPrices(unittest.TestCase):
    def test_official_cost_formula_and_latency_ms(self):
        recs = [_row("gsm8k", 0, "cheap", 1.0, input_tokens=10, output_tokens=20, response_time=0.25)]
        recs.append(_row("gsm8k", 0, "strong", 0.0, input_tokens=10, output_tokens=20, response_time=1.0))
        rows, _ = map_routing_records(recs, dataset="x", prices=_prices())
        by = {(r.query_id, r.model): r for r in rows}
        cheap = by[("0", "cheap")]
        # 10 * 1 / 1e6 + 20 * 2 / 1e6
        self.assertAlmostEqual(cheap.realized_cost, 50.0 / 1e6)
        self.assertEqual(cheap.input_tokens, 10)
        self.assertEqual(cheap.output_tokens, 20)
        self.assertAlmostEqual(cheap.latency_ms, 250.0)
        self.assertIsNone(cheap.router_score)
        self.assertIsNone(cheap.router_assignment)

    def test_token_num_is_not_used_to_invent_splits(self):
        recs = [
            {
                "task_name": "gsm8k",
                "task_id": 1,
                "model_name": "cheap",
                "performance": 1.0,
                "token_num": 400,
                "response_time": 0.1,
            },
            {
                "task_name": "gsm8k",
                "task_id": 1,
                "model_name": "strong",
                "performance": 0.0,
                "token_num": 800,
            },
        ]
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        self.assertEqual(meta["n_cost_missing"], 2)
        for r in rows:
            self.assertIsNone(r.input_tokens)
            self.assertIsNone(r.output_tokens)
            self.assertIsNone(r.realized_cost)
        self.assertIsNone(rows[1].latency_ms)

    def test_does_not_use_ecologic_models_json(self):
        recs = [
            _row("gsm8k", 0, "gpt-4o", 1.0),
            _row("gsm8k", 0, "cheap", 0.0),
        ]
        rows, meta = map_routing_records(recs, dataset="x", prices=_prices())
        by = {r.model: r for r in rows}
        self.assertIsNone(by["gpt-4o"].realized_cost)
        self.assertIsNotNone(by["cheap"].realized_cost)
        self.assertEqual(meta["n_cost_missing"], 1)

    def test_coverage_marks_absent_router_fields(self):
        recs = _balanced_panel(2)
        panel, info = routing_panel_from_records(
            recs, dataset="x", prices=_prices(), min_queries=2
        )
        self.assertIsNotNone(panel)
        cov = info["coverage"]["canonical"]
        self.assertEqual(cov["router_score"]["n_present"], 0)
        self.assertEqual(cov["router_assignment"]["n_present"], 0)
        self.assertTrue(info["coverage"]["embedding_id_not_used_as_primary_id"])
        self.assertTrue(info["coverage"]["token_num_not_used_for_cost"])
        self.assertFalse(info["coverage"]["ecologic_models_json_used"])
        src = info["coverage"]["source_columns"]["embedding_id"]
        self.assertTrue(src["column_present"])

    def test_skips_one_model_per_query_panel(self):
        recs = []
        for i in range(12):
            recs.append(_row("user", i, "cheap" if i % 2 == 0 else "strong", 1.0))
        panel, info = routing_panel_from_records(
            recs, dataset="personal", prices=_prices(), min_queries=10
        )
        self.assertIsNone(panel)
        self.assertEqual(info["status"], "skipped")


class TestAuditAndResume(unittest.TestCase):
    def test_oracle_accounting_and_skipped_router(self):
        recs = _balanced_panel(2)
        job = Job("toy", "test", "all")
        out = run_combo(job, recs, _prices(), n_boot=0, n_grid=4, min_queries=2)
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["usd_available"])
        self.assertEqual(out["n_models"], 2)
        self.assertEqual(out["query_id_scheme"], "task_id")
        self.assertTrue(out["accounting"]["naive_vs_realized"]["reconciles"])
        self.assertIsNotNone(out["relative_accounting_error"])
        self.assertIn("quality_advantage", out["cost_matched_static"]["oracle"])
        self.assertTrue(out["oracle_frontier"]["unconstrained_oracle"]["feasible"])
        self.assertFalse(out["router_frontier"]["available"])
        self.assertIn("no router_score", out["router_frontier"]["reason"])
        rel = relative_accounting_error(2.0, 1.0)
        self.assertTrue(rel["available"])
        self.assertAlmostEqual(rel["relative_accounting_error"], 1.0)

    def test_resume_skips_completed_combo(self):
        subdir = "xroutebench/_unit_resume"
        dest = RESULTS / subdir
        if dest.exists():
            shutil.rmtree(dest)
        key = Job("toy", "test", "all").key
        write_result(
            f"{subdir}/combos/{key}.json",
            {
                "schema_version": "1.0",
                "status": "ok",
                "combo_key": key,
                "config": "toy",
                "split": "test",
                "subset": "all",
                "n_queries": 2,
                "n_models": 2,
                "usd_available": True,
                "paid_model_apis": False,
            },
        )
        calls = []

        def load_split(repo, config, split, cache=None):
            calls.append((config, split))
            raise AssertionError("resume should not reload a completed combo")

        catalog = [
            {
                "name": "toy",
                "kind": "routing",
                "features": ["task_id", "model_name", "performance"],
                "splits": ["test"],
            }
        ]
        try:
            out = run_audit(
                repo="org/demo",
                cache=Path("/tmp/xroutebench_unit_cache"),
                result_subdir=subdir,
                resume=True,
                catalog=catalog,
                prices=_prices(),
                load_split=load_split,
                jobs=[Job("toy", "test", "all")],
                n_boot=0,
                min_queries=2,
            )
            self.assertEqual(calls, [])
            self.assertEqual(out["n_audited_this_process"], 0)
            self.assertEqual(out["n_combos"], 1)
            self.assertTrue(out["combos"][0]["resume_skipped"])
        finally:
            if dest.exists():
                shutil.rmtree(dest)

    def test_combo_complete_helper(self):
        self.assertTrue(combo_is_complete({"schema_version": "1.0", "status": "skipped"}))
        self.assertFalse(combo_is_complete({"status": "ok"}))
        self.assertFalse(combo_is_complete({"schema_version": "1.0", "status": "error"}))

    def test_runner_source_has_no_paid_apis(self):
        from woais_experiments.external import run_xroutebench as mod
        from woais_experiments.external import xroutebench as load_mod

        for src in (inspect.getsource(mod), inspect.getsource(load_mod)):
            self.assertNotIn("chat(", src)
            self.assertNotIn("TOGETHER_API_KEY", src)
            self.assertNotIn("OPENAI_API_KEY", src)
            self.assertNotIn("run_benchmark", src)

    def test_too_many_tasks_are_capped_not_dropped(self):
        from woais_experiments.external.run_xroutebench import (
            MAX_TASK_SUBSETS,
            Job,
            subset_jobs_from_meta,
        )
        meta = {"task_name_query_counts": {f"t{i:02d}": 10 for i in range(MAX_TASK_SUBSETS + 5)}}
        jobs = subset_jobs_from_meta(meta, Job("cfg", "test", "all"))
        self.assertEqual(len(jobs), MAX_TASK_SUBSETS)
        self.assertEqual(jobs[0].subset, "t00")

    def test_complete_case_truncation_is_alphabetical(self):
        from woais_experiments.external.schema import CanonicalRow
        from woais_experiments.external.xroutebench import select_complete_rows
        rows = []
        for q in ("q1", "q2"):
            for m, cov in (("zeta", 1), ("alpha", 1), ("mu", 1)):
                rows.append(
                    CanonicalRow(
                        query_id=q, dataset="d", model=m, quality=1.0,
                        input_tokens=1, output_tokens=1, realized_cost=1.0,
                        latency_ms=1.0,
                    )
                )
        selected, notes = select_complete_rows(rows, require_cost=True, max_models=2, min_models=2, min_queries=2)
        models = sorted({r.model for r in selected})
        self.assertEqual(models, ["alpha", "mu"])
        self.assertTrue(any("by name" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
