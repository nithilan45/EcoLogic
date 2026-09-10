"""Regression tests for the final adversarial audit fixes."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.deployment.analyze_deployment import summarize_group
from woais_experiments.deployment.infer import AppContext, call_with_retries
from woais_experiments.deployment.lifecycle import ProcessLifecycle
from woais_experiments.deployment.provider import RetryableProviderError, StubProvider
from woais_experiments.external.adapter import discover_sources
from woais_experiments.external.schema import QUERY_ALIASES, SchemaError, as_float
from woais_experiments.frozen import matrix_from_calls, to_jsonable
from woais_experiments.paths import RESULTS, public_relpath
from woais_experiments.routing.policies import always_tier, evaluate_policies
from woais_experiments.routing.robustness import score_hull
from woais_experiments.stress.analyze_stress import cell_metrics, pareto_under_load
from woais_experiments.stress.failure_analysis import classify_row


def _toy_matrix():
    calls = []
    for item in ("a", "b"):
        for t in (1, 2, 3):
            calls.append({
                "item_id": item,
                "tier": t,
                "correct": t >= 2,
                "usd": 0.01 * t,
                "latency_s": 0.2,
                "total_tokens": 10,
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "benchmark": "x",
            })
    return matrix_from_calls(calls)


class TestSkipCloneIsNotAPass(unittest.TestCase):
    def test_skipped_clone_is_critical(self):
        from woais_experiments.reproducibility.validate import run_validate_artifact

        src = inspect.getsource(run_validate_artifact)
        self.assertIn("clean_clone_skipped", src)
        self.assertNotIn('"ok": True, "skipped": True', src.replace(" ", ""))


class TestSchemaBooleansAndIds(unittest.TestCase):
    def test_as_float_rejects_bool(self):
        with self.assertRaises(SchemaError):
            as_float(True)
        with self.assertRaises(SchemaError):
            as_float(False)

    def test_generic_id_is_not_a_query_alias(self):
        self.assertNotIn("id", QUERY_ALIASES)


class TestOracleExcludedFromMcnemarFdr(unittest.TestCase):
    def test_oracle_is_labeled_and_out_of_bh_family(self):
        matrix = _toy_matrix()
        policies = {
            "ecologic": always_tier(matrix.item_ids, 1),
            "always_t2": always_tier(matrix.item_ids, 2),
            "oracle_usd": always_tier(matrix.item_ids, 2),
        }
        out = evaluate_policies(matrix, policies, lambda t, i: matrix.usd[(t, i)])
        self.assertEqual(out["policies"]["oracle_usd"]["role"], "hindsight_oracle")
        self.assertEqual(
            out["mcnemar_vs"]["oracle_usd"]["bh_family"],
            "upper_bound_excluded_from_fdr",
        )
        self.assertIsNone(out["mcnemar_vs"]["oracle_usd"]["p_adjusted"])
        self.assertEqual(
            out["mcnemar_vs"]["always_t2"]["bh_family"],
            "mcnemar_vs_ecologic_deployable",
        )
        self.assertIsNotNone(out["mcnemar_vs"]["always_t2"]["p_adjusted"])


class TestQualityPermIsNotHullTest(unittest.TestCase):
    def test_score_hull_labels_the_wrong_estimand(self):
        n = 6
        quality = np.ones((n, 3))
        cost = np.array([[1.0, 2.0, 10.0]] * n)
        choice = np.array([1] * n)
        scored = score_hull(("t1", "t2", "t3"), quality, cost, choice, t2_index=1, n_perm=0)
        self.assertEqual(scored["p_raw_estimand"], "paired_quality_vs_always_t2")
        self.assertFalse(scored["primary_claim_tested_by_p_raw"])


class TestAdapterPathsAreRelative(unittest.TestCase):
    def test_discovered_paths_are_repo_relative(self):
        found = {s.name: s for s in discover_sources()}
        path = found["routellm_s9_committed"].path
        self.assertIsNotNone(path)
        self.assertFalse(str(path).startswith("/Users/"))
        self.assertFalse(str(path).startswith("/home/"))
        self.assertTrue(str(path).startswith("stage7_10/"))


class TestUsageMissingIsNotZeroUsd(unittest.TestCase):
    def test_missing_usage_is_null(self):
        from woais_experiments.paths import ensure_legacy_imports

        ensure_legacy_imports()
        try:
            from api import usage_and_cost
        except Exception as exc:
            self.skipTest(f"benchmark.api not importable: {exc}")
        got = usage_and_cost("openai/gpt-oss-20b", {"choices": [{"finish_reason": "stop"}]})
        self.assertIsNone(got["usd"])
        self.assertIsNone(got["prompt_tokens"])
        self.assertFalse(got["usage_complete"])


class TestAnalyzeLoadDoesNotZeroFill(unittest.TestCase):
    def test_source_skips_missing_usd(self):
        from woais_experiments.paths import ensure_legacy_imports

        ensure_legacy_imports()
        import analyze as legacy_analyze

        src = inspect.getsource(legacy_analyze.load)
        self.assertNotIn("or 0.0", src)
        self.assertIn("if tok is None or usd_v is None", src)


class TestRetryTaxonomyAndCostScope(unittest.TestCase):
    def test_429_is_not_http_5xx(self):
        from woais_experiments.deployment.provider import PaidChatProvider

        src = inspect.getsource(PaidChatProvider.complete)
        self.assertIn('error_type="http_429"', src)
        self.assertNotIn("status in {408, 429}", src)

    def test_retryable_includes_429(self):
        class Boom:
            def complete(self, **kwargs):
                raise RetryableProviderError("rate", http_status=429, error_type="http_429")

        result, retries, err, status = call_with_retries(
            lambda: Boom().complete(), max_retries=0
        )
        self.assertIsNone(result)
        self.assertEqual(err, "http_429")
        self.assertEqual(status, 429)
        self.assertEqual(classify_row({"error_type": "http_429", "http_status": 429}), "http_429")


class TestDeploymentLatencySuccessConditioned(unittest.TestCase):
    def test_failed_requests_do_not_enter_primary_p95(self):
        rows = [
            {
                "end_to_end_ms": 10.0,
                "http_status": 200,
                "error_type": None,
                "realized_provider_cost": 0.01,
                "router_decision_ms": 1.0,
                "environment_kind": "local_stub",
                "cloud_measured": False,
                "setup": "C",
                "setup_name": "ecologic",
                "concurrency": 1,
                "generation_mode": "dry_run",
                "backend": "local",
            },
            {
                "end_to_end_ms": 5000.0,
                "http_status": 504,
                "error_type": "timeout",
                "realized_provider_cost": None,
                "router_decision_ms": 1.0,
                "environment_kind": "local_stub",
                "cloud_measured": False,
                "setup": "C",
                "setup_name": "ecologic",
                "concurrency": 1,
                "generation_mode": "dry_run",
                "backend": "local",
            },
        ]
        g = summarize_group(rows, n_boot=0, seed=1)
        self.assertEqual(g["latency_population"], "successful_requests")
        self.assertAlmostEqual(g["p95_latency_ms"], 10.0)
        self.assertGreater(g["p95_latency_ms_including_failures"], 10.0)
        self.assertTrue(g["not_a_cloud_measurement"])
        self.assertFalse(g["cloud_measured"])


class TestStubIsNotCloud(unittest.TestCase):
    def test_dry_run_context_is_local_stub(self):
        from woais_experiments.deployment.config import load_deployment_config
        from woais_experiments.deployment.pricing import load_deployment_prices

        cfg = load_deployment_config()
        ctx = AppContext(
            allow_api=False,
            dry_run=True,
            config=cfg,
            lifecycle=ProcessLifecycle(),
            prices=load_deployment_prices(cfg),
            provider=StubProvider(sleep_s=0.0),
            models={"tier1": {"name": "m", "provider": "together"}},
            classify_fn=lambda p: type("R", (), {"tier": 1, "reason": "t"})(),
        )
        self.assertEqual(ctx.environment_kind, "local_stub")
        self.assertFalse(ctx.cloud_measured)


class TestStressSuccessLatency(unittest.TestCase):
    def test_timeouts_are_excluded_from_primary_p95(self):
        rows = [
            {
                "measurement_type": "SIMULATED",
                "request_id": "a",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1.0,
                "scheduled_arrival_s": 0.0,
                "complete_s": 0.1,
                "end_to_end_s": 0.1,
                "http_status": 200,
                "error_type": None,
                "service_time_s": 0.1,
            },
            {
                "measurement_type": "SIMULATED",
                "request_id": "b",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1.0,
                "scheduled_arrival_s": 0.2,
                "complete_s": 5.2,
                "end_to_end_s": 5.0,
                "http_status": 504,
                "error_type": "timeout",
                "service_time_s": 5.0,
            },
        ]
        cell = cell_metrics(
            rows,
            measurement_type="SIMULATED",
            sla={"p95_latency_s": 2.0, "failure_rate": 0.05},
            capacity=1,
            strong_model="strong",
            profile="poisson",
            multiplier=1.0,
            policy="ecologic",
        )
        self.assertEqual(cell["latency_population"], "successful_requests")
        self.assertAlmostEqual(cell["p95_latency_s"], 0.1)
        self.assertGreater(cell["p95_latency_s_including_failures"], 1.0)


class TestRequirementsDeclarePyYAML(unittest.TestCase):
    def test_requirements_txt_lists_pyyaml(self):
        from woais_experiments.paths import PACKAGE

        text = (PACKAGE / "requirements.txt").read_text()
        self.assertIn("PyYAML", text)


class TestArtifactPathsAreNotHomeAbsolute(unittest.TestCase):
    def test_to_jsonable_converts_repo_paths_and_redacts_foreign_homes(self):
        inside = str(RESULTS / "ablations" / "index.json")
        converted = to_jsonable(inside)
        self.assertFalse(str(converted).startswith("/Users/"))
        self.assertFalse(str(converted).startswith("/home/"))
        self.assertTrue(str(converted).startswith("woais_experiments/results/"))
        fake_home = chr(47).join(["", "Users", "example_user", "secret.json"])
        redacted = to_jsonable(fake_home)
        self.assertEqual(redacted, "<redacted-absolute>")
        self.assertEqual(public_relpath(RESULTS / "summary.json"), "woais_experiments/results/summary.json")


class TestStressCostDoesNotTreatMissingAsZero(unittest.TestCase):
    def test_missing_cost_is_dropped_from_mean_not_zero_filled(self):
        rows = [
            {
                "measurement_type": "SIMULATED",
                "request_id": "a",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1.0,
                "scheduled_arrival_s": 0.0,
                "complete_s": 0.1,
                "end_to_end_s": 0.1,
                "http_status": 200,
                "error_type": None,
                "service_time_s": 0.1,
                "cost_usd": 1.0,
            },
            {
                "measurement_type": "SIMULATED",
                "request_id": "b",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1.0,
                "scheduled_arrival_s": 0.2,
                "complete_s": 0.3,
                "end_to_end_s": 0.1,
                "http_status": 200,
                "error_type": None,
                "service_time_s": 0.1,
                "cost_usd": None,
            },
        ]
        cell = cell_metrics(
            rows,
            measurement_type="SIMULATED",
            sla={"p95_latency_s": 2.0, "failure_rate": 0.05},
            capacity=1,
            strong_model="strong",
            profile="poisson",
            multiplier=1.0,
            policy="ecologic",
        )
        self.assertEqual(cell["n_costed"], 1)
        self.assertEqual(cell["n_missing_cost"], 1)
        self.assertAlmostEqual(cell["cost_per_query"], 1.0)
        self.assertAlmostEqual(cell["cost_per_scheduled_missing_as_zero"], 0.5)
        self.assertEqual(cell["cost_per_query_denominator"], "n_requests_with_finite_cost")

    def test_pareto_refuses_missing_cost(self):
        cells = [
            {
                "profile": "poisson",
                "multiplier": 1.0,
                "policy": "always_cheap",
                "cost_per_query": 0.001,
                "quality_mean": 0.7,
                "p95_latency_s": 0.2,
            },
            {
                "profile": "poisson",
                "multiplier": 1.0,
                "policy": "ecologic",
                "cost_per_query": None,
                "quality_mean": 0.9,
                "p95_latency_s": 0.2,
            },
        ]
        report = pareto_under_load(cells)[0]
        self.assertIsNone(report["ecologic_pareto_efficient"])
        self.assertIn("not treated as $0", report["note"])


if __name__ == "__main__":
    unittest.main()
