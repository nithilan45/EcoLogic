"""Latency instrumentation: observed spans only, plus cheap/strong/router comparison."""

from __future__ import annotations

import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.latency.analyze_latency import (
    bootstrap_ci,
    mad,
    mean,
    percentile,
    std,
    summarize,
)
from woais_experiments.latency.latency_frontier import (
    compare_direct_and_routed,
    overhead_pct,
)
from woais_experiments.latency.timing import (
    InferenceTimer,
    breakdown_from_legacy_total,
    breakdown_from_log_row,
    ns_to_ms,
    span_ms,
    time_callable,
)


class TestTimingUnits(unittest.TestCase):
    def test_ns_to_ms(self):
        self.assertAlmostEqual(ns_to_ms(1_000_000), 1.0)
        self.assertAlmostEqual(ns_to_ms(2_500_000), 2.5)
        self.assertIsNone(ns_to_ms(None))

    def test_span_requires_both_marks(self):
        self.assertAlmostEqual(span_ms(0, 5_000_000), 5.0)
        self.assertIsNone(span_ms(0, None))
        self.assertIsNone(span_ms(None, 10))


class TestInferenceTimer(unittest.TestCase):
    def test_direct_call_fills_api_and_e2e_not_ttft(self):
        t = InferenceTimer()
        t.marks["request_start"] = 0
        t.marks["response_end"] = 10_000_000  # 10 ms
        rec = t.to_breakdown(query_id="q", model="cheap", policy="direct_cheap")
        self.assertAlmostEqual(rec.provider_api_ms, 10.0)
        self.assertAlmostEqual(rec.end_to_end_ms, 10.0)
        self.assertIsNone(rec.time_to_first_token_ms)
        self.assertIsNone(rec.generation_ms)
        self.assertIsNone(rec.request_queue_ms)
        self.assertIsNone(rec.router_decision_ms)
        self.assertIn("time_to_first_token_ms", rec.missing)

    def test_router_plus_stream_uses_observed_first_token(self):
        t = InferenceTimer()
        t.marks["router_start"] = 0
        t.marks["router_end"] = 2_000_000
        t.marks["request_start"] = 2_000_000
        t.marks["first_token"] = 5_000_000
        t.marks["last_token"] = 12_000_000
        t.marks["response_end"] = 12_000_000
        rec = t.to_breakdown(policy="router_plus_selected")
        self.assertAlmostEqual(rec.router_decision_ms, 2.0)
        self.assertAlmostEqual(rec.time_to_first_token_ms, 3.0)
        self.assertAlmostEqual(rec.generation_ms, 7.0)
        self.assertAlmostEqual(rec.provider_api_ms, 10.0)
        self.assertAlmostEqual(rec.end_to_end_ms, 12.0)

    def test_queue_only_from_provider_payload(self):
        t = InferenceTimer()
        t.marks["request_start"] = 0
        t.marks["response_end"] = 1_000_000
        t.absorb_provider_payload({"prompt_tokens": 100, "completion_tokens": 50})
        rec = t.to_breakdown()
        self.assertIsNone(rec.request_queue_ms)
        t.absorb_provider_payload({"queue_time_ms": 4.5})
        rec2 = t.to_breakdown()
        self.assertAlmostEqual(rec2.request_queue_ms, 4.5)

    def test_time_callable_non_negative(self):
        result, ms = time_callable(lambda x: x + 1, 3)
        self.assertEqual(result, 4)
        self.assertGreaterEqual(ms, 0.0)


class TestLegacyLogs(unittest.TestCase):
    def test_latency_s_becomes_e2e_only(self):
        rec = breakdown_from_legacy_total(latency_s=1.5, query_id="a", selected_tier=2)
        self.assertAlmostEqual(rec.end_to_end_ms, 1500.0)
        self.assertEqual(rec.source, "legacy_latency_s")
        for name in (
            "router_decision_ms",
            "request_queue_ms",
            "provider_api_ms",
            "time_to_first_token_ms",
            "generation_ms",
        ):
            self.assertIsNone(getattr(rec, name), name)

    def test_does_not_infer_ttft_from_tokens(self):
        rec = breakdown_from_log_row(
            {
                "item_id": "q1",
                "latency_s": 2.0,
                "completion_tokens": 200,
                "prompt_tokens": 50,
                "total_tokens": 250,
            }
        )
        self.assertAlmostEqual(rec.end_to_end_ms, 2000.0)
        self.assertIsNone(rec.time_to_first_token_ms)
        self.assertIsNone(rec.generation_ms)
        self.assertIsNone(rec.provider_api_ms)

    def test_instrumented_fields_win_over_latency_s(self):
        rec = breakdown_from_log_row(
            {
                "query_id": "q",
                "latency_s": 9.0,
                "end_to_end_ms": 12.0,
                "router_decision_ms": 1.5,
            }
        )
        self.assertAlmostEqual(rec.end_to_end_ms, 12.0)
        self.assertAlmostEqual(rec.router_decision_ms, 1.5)
        self.assertIsNone(rec.generation_ms)


class TestSummaries(unittest.TestCase):
    def test_percentiles_nearest_rank(self):
        xs = [10, 20, 30, 40, 50]
        self.assertAlmostEqual(percentile(xs, 50), 30)
        self.assertAlmostEqual(percentile(xs, 90), 50)
        self.assertAlmostEqual(percentile(xs, 95), 50)
        self.assertAlmostEqual(percentile(xs, 0), 10)

    def test_mad_odd_sample(self):
        xs = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(mean(xs), 3.0)
        self.assertAlmostEqual(mad(xs), 1.0)
        self.assertAlmostEqual(std(xs), float(np.std(xs, ddof=1)))

    def test_summarize_drops_missing(self):
        out = summarize([1.0, None, 3.0, float("nan")], n_boot=0)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_missing"], 2)
        self.assertAlmostEqual(out["mean"], 2.0)

    def test_bootstrap_constant_is_degenerate(self):
        ci = bootstrap_ci([5.0, 5.0, 5.0, 5.0], "mean", n_boot=50, seed=1)
        self.assertAlmostEqual(ci["point"], 5.0)
        self.assertAlmostEqual(ci["lo"], 5.0)
        self.assertAlmostEqual(ci["hi"], 5.0)

    def test_bootstrap_mean_covers_true_mean(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci(xs, "mean", n_boot=200, seed=20260909)
        self.assertLessEqual(ci["lo"], 3.0)
        self.assertGreaterEqual(ci["hi"], 3.0)


class TestFrontier(unittest.TestCase):
    def test_three_policy_means(self):
        # q1: cheap 10, strong 100, router picks cheap
        # q2: cheap 20, strong 80, router picks strong
        payload = compare_direct_and_routed(
            ["q1", "q2"],
            [10.0, 20.0],
            [100.0, 80.0],
            [10.0, 80.0],
            selected_tier=[2, 3],
            n_boot=0,
        )
        self.assertAlmostEqual(payload["policies"]["direct_cheap"]["mean"], 15.0)
        self.assertAlmostEqual(payload["policies"]["direct_strong"]["mean"], 90.0)
        self.assertAlmostEqual(payload["policies"]["router_plus_selected"]["mean"], 45.0)
        self.assertEqual(payload["router_overhead"]["n_measured"], 0)
        self.assertIsNone(payload["per_query"][0]["router_overhead_ms"])
        self.assertAlmostEqual(payload["per_query"][0]["router_minus_cheap_ms"], 0.0)
        self.assertAlmostEqual(payload["per_query"][1]["router_minus_cheap_ms"], 60.0)

    def test_overhead_absolute_and_percent(self):
        self.assertAlmostEqual(overhead_pct(2.0, 50.0), 4.0)
        self.assertIsNone(overhead_pct(None, 50.0))
        payload = compare_direct_and_routed(
            ["q"],
            [40.0],
            [200.0],
            [42.0],
            router_decision_ms=[2.0],
            source="instrumented",
            n_boot=0,
        )
        row = payload["per_query"][0]
        self.assertAlmostEqual(row["router_overhead_ms"], 2.0)
        self.assertAlmostEqual(row["router_overhead_pct"], 100.0 * 2.0 / 42.0)
        self.assertEqual(payload["router_overhead"]["n_measured"], 1)
        self.assertAlmostEqual(payload["router_overhead"]["absolute_ms"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
