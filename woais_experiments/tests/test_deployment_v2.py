"""MEASURED_REAL_DEPLOYMENT (deployment_real_v2). Does not call paid APIs."""

from __future__ import annotations

import inspect
import json
import shutil
import time
import unittest
from pathlib import Path

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.deployment.analyze_real import (
    FILE_PROVENANCE_KEYS,
    MixedSourceError,
    RESULTS_PREFIX,
    analyze_records,
    summarize_group,
)
from woais_experiments.deployment.budget import abort_if_over_budget, worst_case_budget
from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import build_context, handle_inference
from woais_experiments.deployment.lifecycle import ProcessLifecycle, cold_start_observed, keep_lifecycle_if_same_process
from woais_experiments.deployment.query_set import TARGET_N, select_query_set
from woais_experiments.deployment.records import (
    MEASUREMENT_TYPE,
    MEASUREMENT_TYPE_DRY_RUN,
    MEASUREMENT_TYPE_PAID_LOCAL,
    MEASUREMENT_TYPE_REAL,
    RequestRecord,
    v2_measurement_type,
)
from woais_experiments.deployment.run_real import (
    REAL_COMMAND,
    SafetyError,
    build_parser,
    enforce_safety,
    persist,
    run_real,
)
from woais_experiments.deployment.static_mix import cheap_strong_slugs
from woais_experiments.paths import RESULTS, reset_run_context, set_run_context
from woais_experiments.runner.session import COMMANDS, STAGE_ORDER
from woais_experiments.stress.load_generator import CostCapExceeded


def _tmp_root() -> Path:
    root = RESULTS / "_deploy_v2_unittest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _tiny_cfg() -> dict:
    cfg = load_deployment_config()
    cfg["listen"] = {"host": "127.0.0.1", "port": 0}
    cfg["stub"] = {"sleep_s": 0.0, "output_tokens": 16, "chars_per_token": 4.0}
    cfg["bootstrap"] = {"n_boot": 40, "seed": 1, "level": 0.95}
    return cfg


class TestSafetyFlags(unittest.TestCase):
    def test_paid_requires_all_three_flags(self):
        with self.assertRaises(SafetyError) as ctx:
            enforce_safety(
                allow_api=True, allow_cloud=False, max_cost_usd=1.0,
                dry_run=False, backend="local", base_url=None,
            )
        self.assertIn("--allow-cloud", str(ctx.exception))
        with self.assertRaises(SafetyError) as ctx:
            enforce_safety(
                allow_api=True, allow_cloud=True, max_cost_usd=None,
                dry_run=False, backend="local", base_url=None,
            )
        self.assertIn("--max-cost-usd", str(ctx.exception))
        self.assertIn("no default", str(ctx.exception).lower())

    def test_parser_has_no_default_spend_limit(self):
        action = next(a for a in build_parser()._actions if a.dest == "max_cost_usd")
        self.assertIsNone(action.default)
        from woais_experiments.runner.cli import build_parser as woais_parser
        action = next(a for a in woais_parser()._actions if a.dest == "max_cost_usd")
        self.assertIsNone(action.default)
        args = woais_parser().parse_args(["deployment-real"])
        self.assertIsNone(args.max_cost_usd)
        self.assertFalse(args.allow_api)
        self.assertFalse(args.allow_cloud)

    def test_cli_refuses_incomplete_paid_flags(self):
        from woais_experiments.runner.cli import main
        self.assertEqual(main(["deployment-real", "--allow-api"]), 2)
        self.assertEqual(main(["deployment-real", "--allow-api", "--allow-cloud"]), 2)
        self.assertEqual(main(["deployment-real", "--allow-api", "--max-cost-usd", "1"]), 2)

    def test_command_is_registered_and_not_a_stage(self):
        self.assertIn("deployment-real", COMMANDS)
        self.assertNotIn("deployment-real", STAGE_ORDER)

    def test_loopback_is_not_serverless(self):
        paid, serverless = enforce_safety(
            allow_api=True, allow_cloud=True, max_cost_usd=5.0,
            dry_run=False, backend="cloudrun", base_url="http://127.0.0.1:8080",
        )
        self.assertTrue(paid)
        self.assertFalse(serverless)


class TestBudgetAbort(unittest.TestCase):
    def test_worst_case_over_cap_aborts_before_inference(self):
        root = _tmp_root()
        tokens = set_run_context(output_root=root, overwrite_policy="replace")
        cfg = _tiny_cfg()
        try:
            ctx = build_context(cfg, allow_api=False, dry_run=True, backend="local")
            _cheap, strong = cheap_strong_slugs(ctx.models, ctx.roles)
            estimate = worst_case_budget(
                [{"text": "hello world"}],
                prices=ctx.prices,
                strong_model=strong,
                n_policies=4,
                n_workloads=4,
                max_output_tokens=64,
                max_retries=2,
            )
            with self.assertRaises(CostCapExceeded):
                abort_if_over_budget(estimate, 1e-12)
            with self.assertRaises(CostCapExceeded):
                run_real(
                    cfg,
                    allow_api=True,
                    allow_cloud=True,
                    max_cost_usd=1e-12,
                    dry_run=False,
                    backend="local",
                    base_url=None,
                    n_queries=4,
                    max_queries=2,
                    include_concurrency_8=False,
                    idle_s=0.0,
                    prefix=RESULTS_PREFIX,
                    write_preflight=True,
                )
            qs = root / RESULTS_PREFIX / "query_set.json"
            self.assertTrue(qs.exists(), "query IDs must be stored before any inference")
            payload = json.loads(qs.read_text())
            self.assertTrue(payload["frozen_before_benchmark"])
            self.assertEqual(payload["measurement_type"], MEASUREMENT_TYPE_PAID_LOCAL)
            self.assertFalse((root / RESULTS_PREFIX / "requests.jsonl").exists())
        finally:
            reset_run_context(tokens)
            shutil.rmtree(root, ignore_errors=True)


class TestQuerySetAndColdStart(unittest.TestCase):
    def test_official_query_set_is_fixed_before_outcomes(self):
        a = select_query_set(n=TARGET_N)
        b = select_query_set(n=TARGET_N)
        self.assertEqual(a["n"], TARGET_N)
        self.assertEqual(a["query_ids"], b["query_ids"])
        self.assertEqual(a["query_set_hash"], b["query_set_hash"])
        self.assertTrue(a["frozen_before_benchmark"])
        self.assertGreaterEqual(a["n"], 50)
        self.assertLessEqual(a["n"], 150)

    def test_cold_start_is_process_init_not_latency(self):
        life = ProcessLifecycle()
        first = life.observe()
        time.sleep(0.02)
        second = life.observe()
        self.assertTrue(cold_start_observed(first))
        self.assertFalse(cold_start_observed(second))
        self.assertNotIn("latency", inspect.signature(cold_start_observed).parameters)
        ctx = build_context(_tiny_cfg(), allow_api=False, dry_run=True, backend="local")
        _, rec1 = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        _, rec2 = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertTrue(rec1.cold_start_observed)
        self.assertFalse(rec2.cold_start_observed)
        rec2.end_to_end_ms = 50_000.0
        self.assertFalse(rec2.cold_start_observed)

    def test_same_pid_recycle_is_not_a_cold_start(self):
        ctx = build_context(_tiny_cfg(), allow_api=False, dry_run=True, backend="local")
        handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        ctx.lifecycle = keep_lifecycle_if_same_process(ctx.lifecycle)
        _, rec = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertFalse(rec.cold_start_observed)

    def test_v2_measurement_type_mapping(self):
        self.assertEqual(v2_measurement_type(paid=False, serverless=False), MEASUREMENT_TYPE_DRY_RUN)
        self.assertEqual(v2_measurement_type(paid=True, serverless=False), MEASUREMENT_TYPE_PAID_LOCAL)
        self.assertEqual(v2_measurement_type(paid=True, serverless=True), MEASUREMENT_TYPE_REAL)
        self.assertEqual(v2_measurement_type(paid=False, serverless=True), MEASUREMENT_TYPE_DRY_RUN)


class TestAnalyzeReal(unittest.TestCase):
    def test_refuses_simulated_and_energy(self):
        with self.assertRaises(MixedSourceError):
            analyze_records([{"measurement_type": "SIMULATED", "end_to_end_ms": 1, "setup": "C"}])
        row = RequestRecord(
            request_id="x",
            timestamp="t",
            query_length_chars=1,
            input_tokens=1,
            router_decision_ms=0.1,
            provider_request_ms=1.0,
            time_to_first_token_ms=None,
            generation_ms=None,
            end_to_end_ms=1.0,
            selected_model="gpt-4o",
            output_tokens=1,
            realized_provider_cost=0.0,
            http_status=200,
            error_type=None,
            retry_count=0,
            measurement_type=MEASUREMENT_TYPE_REAL,
            setup="C",
            setup_name="ecologic",
            workload="sequential_warm",
        ).to_dict()
        row["energy_j"] = 1.0
        with self.assertRaises(MixedSourceError):
            analyze_records([row])
        row.pop("energy_j")
        summary = analyze_records([row], n_boot=20, seed=1)
        self.assertEqual(summary["measurement_type"], MEASUREMENT_TYPE_REAL)
        self.assertFalse(summary["energy"]["measured"])
        g = summarize_group([row], n_boot=20, seed=1)
        self.assertFalse(g["energy"]["measured"])
        self.assertEqual(g["cold_start_count"], 0)

    def test_v1_measurement_type_is_rejected_here(self):
        with self.assertRaises(MixedSourceError):
            analyze_records([{"measurement_type": MEASUREMENT_TYPE, "end_to_end_ms": 1}])

    def test_infeasible_static_mix_is_not_cost_matched(self):
        def _row(name: str, cost: float) -> dict:
            rec = RequestRecord(
                request_id=name,
                timestamp="t",
                query_length_chars=1,
                input_tokens=1,
                router_decision_ms=0.1,
                provider_request_ms=1.0,
                time_to_first_token_ms=None,
                generation_ms=None,
                end_to_end_ms=1.0,
                selected_model="gpt-4o",
                output_tokens=1,
                realized_provider_cost=cost,
                http_status=200,
                error_type=None,
                retry_count=0,
                measurement_type=MEASUREMENT_TYPE_DRY_RUN,
                setup="C" if name == "eco" else "D",
                setup_name="ecologic" if name == "eco" else "cost_matched_static",
                workload="burst",
            ).to_dict()
            return rec

        summary = analyze_records(
            [_row("eco", 0.01), _row("static", 0.02)],
            n_boot=20,
            seed=1,
            static_mix={"cost_match_feasible": False, "mix_fraction_strong": 0.0},
        )
        self.assertEqual(summary["measurement_type"], MEASUREMENT_TYPE_DRY_RUN)
        cmp = summary["comparison_ecologic_vs_cost_matched_static"]
        self.assertFalse(cmp["cost_match_feasible"])
        self.assertEqual(len(cmp["workloads"]), 1)
        self.assertIsNone(cmp["workloads"][0]["cost_savings_vs_static"])
        self.assertEqual(cmp["workloads"][0]["comparator_role"], "clamped_static_not_cost_matched")
        self.assertEqual(
            cmp["workloads"][0]["pareto_status_cost_vs_p95_latency"],
            "not_a_cost_matched_comparison",
        )


class TestTinyLocalDryRun(unittest.TestCase):
    def test_same_ids_all_policies_local_not_serverless(self):
        root = _tmp_root()
        tokens = set_run_context(output_root=root, overwrite_policy="replace")
        cfg = _tiny_cfg()
        try:
            result = run_real(
                cfg,
                allow_api=False,
                allow_cloud=False,
                max_cost_usd=None,
                dry_run=True,
                backend="local",
                base_url=None,
                n_queries=4,
                max_queries=2,
                include_concurrency_8=False,
                idle_s=0.0,
                prefix=RESULTS_PREFIX,
            )
            self.assertEqual(result["generation_mode"], "dry_run")
            self.assertFalse(result["paid"])
            self.assertFalse(result["serverless"])
            self.assertEqual(result["query_set"]["n"], 4)
            self.assertEqual(result["query_set"]["executed_n"], 2)
            ids = result["query_set"]["executed_ids"]
            self.assertEqual(len(ids), 2)
            by_policy: dict[str, set[str]] = {}
            for row in result["records"]:
                self.assertEqual(row["measurement_type"], MEASUREMENT_TYPE_DRY_RUN)
                self.assertFalse(row["serverless"])
                self.assertNotEqual(row.get("environment_kind"), "cloudrun")
                self.assertFalse(row.get("cloud_measured"))
                self.assertIn("arrival_timestamp", row)
                self.assertIn("cold_start_observed", row)
                self.assertIn("instance_id", row)
                self.assertIn("provider_cost", row)
                self.assertIn("provider_latency_ms", row)
                self.assertIn("git_commit", row)
                self.assertIn("pricing_config_hash", row)
                self.assertIn("query_set_hash", row)
                self.assertFalse(row["energy_measured"])
                by_policy.setdefault(row["setup_name"], set()).add(row["query_id"])
            # 4 policies × 4 workloads × 2 queries
            self.assertEqual(len(result["records"]), 32)
            self.assertEqual(set(by_policy), {"always_cheap", "always_strong", "ecologic", "cost_matched_static"})
            for name, qids in by_policy.items():
                self.assertEqual(qids, set(ids), msg=name)
            seq = [r for r in result["records"] if r["workload"] == "sequential_warm" and r["setup"] == "A"]
            self.assertTrue(any(r.get("cold_start_observed") for r in seq))
            idle = [r for r in result["records"] if r["workload"] == "idle_then_batch" and r["setup"] == "A"]
            self.assertTrue(idle)
            self.assertFalse(any(r.get("cold_start_observed") for r in idle))
            paths = persist(result, prefix=RESULTS_PREFIX)
            self.assertIn("summary_json", paths)
            self.assertIn("requests_jsonl", paths)
            self.assertTrue((root / RESULTS_PREFIX / "requests.jsonl").exists())
            self.assertTrue((root / RESULTS_PREFIX / "summary.json").exists())
            self.assertTrue((root / RESULTS_PREFIX / "query_set.json").exists())
            summary = json.loads((root / RESULTS_PREFIX / "summary.json").read_text())
            self.assertEqual(summary["measurement_type"], MEASUREMENT_TYPE_DRY_RUN)
            self.assertFalse(summary["energy"]["measured"])
            cmp = summary["comparison_ecologic_vs_cost_matched_static"]
            if summary.get("static_mix", {}).get("cost_match_feasible") is False or result["mix"].get("cost_match_feasible") is False:
                self.assertFalse(cmp.get("cost_match_feasible"))
                for row in cmp.get("workloads") or []:
                    self.assertIsNone(row.get("cost_savings_vs_static"))
                    self.assertEqual(row.get("comparator_role"), "clamped_static_not_cost_matched")
            for key in FILE_PROVENANCE_KEYS:
                self.assertIn(key, summary)
            self.assertIn("--allow-api", REAL_COMMAND)
            self.assertIn("--allow-cloud", REAL_COMMAND)
            self.assertIn("--max-cost-usd", REAL_COMMAND)
            self.assertTrue((root / RESULTS_PREFIX / "summary.json").exists())
            self.assertFalse((root / "deployment_real").exists())
        finally:
            reset_run_context(tokens)
            shutil.rmtree(root, ignore_errors=True)

    def test_idle_http_restart_does_not_fabricate_cold_start(self):
        root = _tmp_root()
        tokens = set_run_context(output_root=root, overwrite_policy="replace")
        cfg = _tiny_cfg()
        try:
            result = run_real(
                cfg,
                allow_api=False,
                allow_cloud=False,
                max_cost_usd=None,
                dry_run=True,
                backend="local",
                base_url=None,
                n_queries=4,
                max_queries=2,
                include_concurrency_8=False,
                idle_s=0.05,
                prefix=RESULTS_PREFIX,
            )
            idle = [
                r for r in result["records"]
                if r["workload"] == "idle_then_batch" and r["setup"] == "A"
            ]
            self.assertTrue(idle)
            self.assertFalse(any(r.get("cold_start_observed") for r in idle))
            pids = {r.get("process_id") for r in result["records"] if r.get("process_id") is not None}
            self.assertEqual(len(pids), 1)
        finally:
            reset_run_context(tokens)
            shutil.rmtree(root, ignore_errors=True)


class TestCliSource(unittest.TestCase):
    def test_cli_lazy_imports_run_real(self):
        from woais_experiments.runner import cli
        src = inspect.getsource(cli)
        self.assertIn("deployment-real", src)
        self.assertNotIn("httpx", src)
        self.assertNotIn("chat(", src)


if __name__ == "__main__":
    unittest.main()
