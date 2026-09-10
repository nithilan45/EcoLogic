"""Deployment benchmark: timing, cost, logging, retries, cold/warm, malformed."""

from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.deployment.analyze_deployment import (
    analyze_records,
)
from woais_experiments.deployment.app import lambda_handler, reset_lambda_context, start_server, stop_server
from woais_experiments.deployment.client import DeploymentClient
from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import (
    AppContext,
    build_context,
    call_with_retries,
    handle_inference,
)
from woais_experiments.deployment.lifecycle import BASIS_FIRST, BASIS_SUBSEQUENT, COLD, WARM, ProcessLifecycle
from woais_experiments.deployment.pricing import load_deployment_prices
from woais_experiments.deployment.provider import (
    PaidApiRefused,
    PaidChatProvider,
    RetryableProviderError,
    StubProvider,
    parse_chat_completion,
)
from woais_experiments.deployment.records import MEASUREMENT_TYPE, REQUIRED_REQUEST_FIELDS, RequestRecord, assert_measured_record
from woais_experiments.deployment.router import classify_prompt, load_production_router


class FailNTimes:
    def __init__(self, n: int, inner: StubProvider):
        self.n = n
        self.left = n
        self.inner = inner
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            raise RetryableProviderError("injected", http_status=503, error_type="http_5xx")
        return self.inner.complete(**kwargs)


class MalformedProvider:
    def complete(self, **kwargs):
        del kwargs
        return parse_chat_completion("not-an-object", http_status=200)


def _ctx(**kwargs) -> AppContext:
    cfg = load_deployment_config()
    cfg.setdefault("stub", {})["sleep_s"] = kwargs.pop("sleep_s", 0.002)
    cfg.setdefault("retries", {})["max_retries"] = kwargs.pop("max_retries", 2)
    provider = kwargs.pop("provider", None)
    lifecycle = kwargs.pop("lifecycle", None)
    return build_context(
        cfg,
        allow_api=False,
        dry_run=True,
        backend="local",
        provider=provider,
        lifecycle=lifecycle,
        **kwargs,
    )


class TestTimingInstrumentation(unittest.TestCase):
    def test_router_and_provider_spans_are_observed(self):
        ctx = _ctx(sleep_s=0.01)
        status, rec = handle_inference(
            {"prompt": "What is photosynthesis?", "policy": "ecologic"},
            ctx=ctx,
        )
        self.assertEqual(status, 200)
        self.assertIsNotNone(rec.router_decision_ms)
        self.assertGreaterEqual(rec.router_decision_ms, 0.0)
        self.assertIsNotNone(rec.provider_request_ms)
        self.assertGreaterEqual(rec.provider_request_ms, 8.0)
        self.assertIsNotNone(rec.end_to_end_ms)
        self.assertGreaterEqual(rec.end_to_end_ms, rec.provider_request_ms)
        self.assertIsNone(rec.time_to_first_token_ms)
        self.assertIsNone(rec.generation_ms)
        self.assertEqual(rec.measurement_type, MEASUREMENT_TYPE)

    def test_direct_cheap_does_not_impute_router_time(self):
        ctx = _ctx()
        _, rec = handle_inference(
            {"prompt": "What is photosynthesis?", "policy": "direct_cheap"},
            ctx=ctx,
        )
        self.assertIsNone(rec.router_decision_ms)
        self.assertIsNotNone(rec.provider_request_ms)
        self.assertEqual(rec.selected_model, "google/gemma-3n-E4B-it")


class TestCostCalculation(unittest.TestCase):
    def test_realized_cost_matches_price_table_times_tokens(self):
        cfg = load_deployment_config()
        prices = load_deployment_prices(cfg)
        resolved = prices.resolve("gpt-4o")
        cost = resolved.inference_cost(100, 20)
        expected = 100 * (2.50 / 1_000_000) + 20 * (10.00 / 1_000_000)
        self.assertAlmostEqual(cost, expected, places=18)
        self.assertEqual(resolved.source, "models.json")

    def test_product_slug_uses_labeled_eval_stack_alias(self):
        prices = load_deployment_prices(load_deployment_config())
        gemma = prices.resolve("google/gemma-3n-E4B-it")
        self.assertEqual(gemma.price_slug, "Qwen/Qwen3.5-9B")
        self.assertTrue(gemma.source.startswith("eval_stack_proxy:"))
        stub = StubProvider(sleep_s=0, output_tokens=16, chars_per_token=4.0)
        ctx = _ctx(provider=stub)
        prompt = "abcd" * 10  # 40 chars → 10 input tokens
        _, rec = handle_inference({"prompt": prompt, "policy": "direct_cheap"}, ctx=ctx)
        self.assertEqual(rec.input_tokens, 10)
        self.assertEqual(rec.output_tokens, 16)
        got, resolved = ctx.prices.realized_provider_cost(rec.selected_model, 10, 16)
        self.assertAlmostEqual(rec.realized_provider_cost, got, places=18)
        self.assertEqual(rec.cost_basis, "dry_run_token_estimate")
        self.assertIsNotNone(resolved)

    def test_missing_usage_does_not_invent_tokens_or_cost(self):
        payload = {"choices": [{"message": {"content": "hi"}}]}
        result = parse_chat_completion(payload, http_status=200)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)
        prices = load_deployment_prices(load_deployment_config())
        cost, resolved = prices.realized_provider_cost("gpt-4o", result.input_tokens, result.output_tokens)
        self.assertIsNone(cost)
        self.assertIsNone(resolved)


class TestRequestLogging(unittest.TestCase):
    def test_required_fields_are_present(self):
        ctx = _ctx()
        _, rec = handle_inference(
            {"prompt": "What is photosynthesis?", "policy": "ecologic", "request_id": "req-1"},
            ctx=ctx,
        )
        row = rec.to_dict()
        for name in REQUIRED_REQUEST_FIELDS:
            self.assertIn(name, row)
        self.assertEqual(row["measurement_type"], MEASUREMENT_TYPE)
        self.assertEqual(row["request_id"], "req-1")
        self.assertEqual(row["query_length_chars"], len("What is photosynthesis?"))
        assert_measured_record(row)


class TestRetryHandling(unittest.TestCase):
    def test_retries_then_succeeds(self):
        inner = StubProvider(sleep_s=0)
        provider = FailNTimes(2, inner)
        ctx = _ctx(provider=provider, max_retries=3)
        status, rec = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertEqual(status, 200)
        self.assertEqual(rec.retry_count, 2)
        self.assertEqual(provider.calls, 3)
        self.assertIsNone(rec.error_type)

    def test_exhausted_retries_record_error(self):
        inner = StubProvider(sleep_s=0)
        provider = FailNTimes(5, inner)
        ctx = _ctx(provider=provider, max_retries=2)
        status, rec = handle_inference({"prompt": "hello?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertGreaterEqual(status, 500)
        self.assertEqual(rec.retry_count, 2)
        self.assertEqual(rec.error_type, "http_5xx")

    def test_malformed_provider_is_not_retried(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            parse_chat_completion([], http_status=200)

        result, retries, err, status = call_with_retries(fn, max_retries=3)
        self.assertIsNone(result)
        self.assertEqual(retries, 0)
        self.assertEqual(err, "malformed_response")
        self.assertEqual(calls["n"], 1)


class TestColdWarmLabeling(unittest.TestCase):
    def test_first_request_cold_second_warm_same_process(self):
        life = ProcessLifecycle()
        ctx = _ctx(lifecycle=life)
        _, a = handle_inference({"prompt": "What is photosynthesis?", "policy": "ecologic"}, ctx=ctx)
        _, b = handle_inference({"prompt": "What is photosynthesis?", "policy": "ecologic"}, ctx=ctx)
        self.assertEqual(a.lifecycle_state, COLD)
        self.assertEqual(a.lifecycle_basis, BASIS_FIRST)
        self.assertEqual(b.lifecycle_state, WARM)
        self.assertEqual(b.lifecycle_basis, BASIS_SUBSEQUENT)
        self.assertEqual(a.process_id, b.process_id)
        self.assertEqual(b.request_index_in_process, 2)

    def test_idle_gap_does_not_relabel_live_process_as_cold(self):
        life = ProcessLifecycle()
        ctx = _ctx(lifecycle=life, provider=StubProvider(sleep_s=0.02))
        _, a = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        time.sleep(0.05)
        _, b = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertEqual(a.lifecycle_state, COLD)
        self.assertEqual(b.lifecycle_state, WARM)
        self.assertGreater(b.process_uptime_s, a.process_uptime_s)
        self.assertGreaterEqual(b.end_to_end_ms, 10.0)

    def test_new_process_is_cold_again(self):
        ctx1 = _ctx(lifecycle=ProcessLifecycle())
        ctx2 = _ctx(lifecycle=ProcessLifecycle())
        _, a = handle_inference({"prompt": "What is photosynthesis?", "policy": "ecologic"}, ctx=ctx1)
        _, b = handle_inference({"prompt": "What is photosynthesis?", "policy": "ecologic"}, ctx=ctx2)
        self.assertEqual(a.lifecycle_state, COLD)
        self.assertEqual(b.lifecycle_state, COLD)


class TestMalformedResponses(unittest.TestCase):
    def test_non_json_body_is_400(self):
        ctx = _ctx()
        status, rec = handle_inference(None, ctx=ctx, raw_ok=False)
        self.assertEqual(status, 400)
        self.assertEqual(rec.error_type, "malformed_request")
        self.assertEqual(rec.measurement_type, MEASUREMENT_TYPE)

    def test_empty_prompt_is_400(self):
        ctx = _ctx()
        status, rec = handle_inference({"prompt": "   ", "policy": "ecologic"}, ctx=ctx)
        self.assertEqual(status, 400)
        self.assertEqual(rec.error_type, "malformed_request")

    def test_malformed_provider_json(self):
        ctx = _ctx(provider=MalformedProvider())
        status, rec = handle_inference({"prompt": "What is photosynthesis?", "policy": "direct_cheap"}, ctx=ctx)
        self.assertEqual(rec.error_type, "malformed_response")
        self.assertGreaterEqual(status, 400)

    def test_client_flags_non_json_http_body(self):
        class BadHandler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = b"{not json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), BadHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            client = DeploymentClient(f"http://127.0.0.1:{port}", timeout_s=2, max_retries=0)
            rec = client.infer("What is photosynthesis?", policy="ecologic")
            self.assertEqual(rec.error_type, "malformed_response")
            self.assertEqual(rec.measurement_type, MEASUREMENT_TYPE)
            self.assertIsNotNone(rec.end_to_end_ms)
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestPaidApiGuard(unittest.TestCase):
    def test_paid_provider_refuses_without_flag(self):
        p = PaidChatProvider(allow_api=False)
        with self.assertRaises(PaidApiRefused):
            p.complete(model="gpt-4o", provider_name="openai", prompt="hi")

    def test_default_context_uses_stub(self):
        ctx = _ctx()
        self.assertIsInstance(ctx.provider, StubProvider)
        self.assertEqual(ctx.generation_mode, "dry_run")


class TestProductionClassifierWrap(unittest.TestCase):
    def test_loads_backend_source_not_a_reimplementation_copy(self):
        fn, models = load_production_router()
        self.assertIn("tier1", models)
        d = classify_prompt("What is photosynthesis?", models=models, classify_fn=fn)
        self.assertEqual(d.recommended_tier, 1)
        d2 = classify_prompt(
            "Write a Python function to parse json",
            models=models,
            classify_fn=fn,
        )
        self.assertEqual(d2.recommended_tier, 3)


class TestAnalyzeMeasuredOnly(unittest.TestCase):
    def test_refuses_simulator_records(self):
        row = RequestRecord(
            request_id="x",
            timestamp="t",
            query_length_chars=1,
            input_tokens=1,
            router_decision_ms=None,
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
        ).to_dict()
        row["des_wait_s"] = 0.1
        with self.assertRaises(Exception):
            assert_measured_record(row)

    def test_refuses_simulated_measurement_type(self):
        with self.assertRaises(ValueError):
            analyze_records([{"measurement_type": "SIMULATED", "end_to_end_ms": 1}])

    def test_bootstrap_ci_on_latency(self):
        rows = []
        for i, lat in enumerate([10.0, 12.0, 11.0, 40.0, 9.0]):
            rec = RequestRecord(
                request_id=f"r{i}",
                timestamp="t",
                query_length_chars=3,
                input_tokens=1,
                router_decision_ms=0.5,
                provider_request_ms=lat,
                time_to_first_token_ms=None,
                generation_ms=None,
                end_to_end_ms=lat,
                selected_model="gpt-4o",
                output_tokens=1,
                realized_provider_cost=0.001,
                http_status=200,
                error_type=None,
                retry_count=0,
                setup="C",
                setup_name="ecologic",
                concurrency=1,
                generation_mode="dry_run",
                backend="local",
            )
            rec.extra["phase_wall_s"] = 1.0
            rows.append(rec.to_dict())
        summary = analyze_records(rows, n_boot=50, seed=1)
        self.assertEqual(summary["measurement_type"], MEASUREMENT_TYPE)
        g = summary["groups"][0]
        self.assertEqual(g["n"], 5)
        self.assertIn("lo", g["latency_ci"]["mean"])
        self.assertIn("hi", g["latency_ci"]["p95"])
        self.assertIsNotNone(g["router_overhead_ms"])
        self.assertAlmostEqual(g["total_realized_cost"], 0.005, places=12)


class TestHttpEndpointAndLambda(unittest.TestCase):
    def test_local_http_roundtrip(self):
        ctx = _ctx()
        httpd, _t = start_server(ctx, host="127.0.0.1", port=0)
        try:
            port = httpd.server_address[1]
            client = DeploymentClient(f"http://127.0.0.1:{port}", timeout_s=3, max_retries=0)
            health = client.health()
            self.assertTrue(health["ok"])
            rec = client.infer("What is photosynthesis?", policy="ecologic")
            self.assertEqual(rec.http_status, 200)
            self.assertEqual(rec.lifecycle_state, COLD)
            rec2 = client.infer("What is photosynthesis?", policy="ecologic")
            self.assertEqual(rec2.lifecycle_state, WARM)
            self.assertEqual(rec.measurement_type, MEASUREMENT_TYPE)
        finally:
            stop_server(httpd)

    def test_lambda_handler_direct_event(self):
        reset_lambda_context()
        out = lambda_handler({"prompt": "What is photosynthesis?", "policy": "ecologic"}, None)
        self.assertEqual(out["measurement_type"], MEASUREMENT_TYPE)
        body = json.loads(out["body"])
        self.assertEqual(body["measurement_type"], MEASUREMENT_TYPE)
        self.assertEqual(out["statusCode"], 200)
        reset_lambda_context()


class TestDryRunBenchmark(unittest.TestCase):
    def test_tiny_local_dry_run(self):
        from woais_experiments.deployment.benchmark import run_benchmark

        cfg = load_deployment_config()
        cfg["prompts"] = [
            {"id": "a", "text": "What is photosynthesis?"},
            {"id": "b", "text": "Compare REST and GraphQL for APIs."},
        ]
        cfg["workloads"] = {
            "concurrency": [1, 2],
            "max_concurrency": 2,
            "concurrency_16_safe": False,
            "idle_gap_s": 0.0,
        }
        cfg["bootstrap"] = {"n_boot": 40, "seed": 1, "level": 0.95}
        cfg["listen"] = {"host": "127.0.0.1", "port": 0}
        cfg["stub"] = {"sleep_s": 0.0, "output_tokens": 16, "chars_per_token": 4.0}
        result = run_benchmark(cfg, allow_api=False, dry_run=True, backend="local", base_url=None)
        self.assertEqual(result["generation_mode"], "dry_run")
        self.assertEqual(result["summary"]["measurement_type"], MEASUREMENT_TYPE)
        # 4 setups × 2 concurrencies × 2 prompts
        self.assertEqual(len(result["records"]), 16)
        self.assertTrue(all(r["measurement_type"] == MEASUREMENT_TYPE for r in result["records"]))
        modes = {g["setup"] for g in result["summary"]["groups"]}
        self.assertEqual(modes, {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
