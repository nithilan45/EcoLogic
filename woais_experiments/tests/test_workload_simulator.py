"""Discrete-event SIMULATED workload tests. No cloud measurements."""

from __future__ import annotations

import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.workloads.arrival_processes import (
    arrivals_from_config,
    constant_arrivals,
    on_off_arrivals,
    poisson_arrivals,
    trace_replay,
)
from woais_experiments.workloads.run_workload_sweep import expand_ofat, run_sweeps
from woais_experiments.workloads.simulator import (
    catalog_synthetic,
    simulate,
    validate_config,
)


def _mini_cfg(**over) -> dict:
    cfg = {
        "simulated": True,
        "label": "SIMULATED",
        "seed": 1,
        "n_requests": 2,
        "shared_arrivals_across_policies": True,
        "catalog": {
            "source": "synthetic",
            "cheap_model": "cheap",
            "strong_model": "strong",
            "query_ids": ["a", "b"],
            "models": {
                "cheap": {"service_s": 1.0, "cost": 0.001},
                "strong": {"service_s": 4.0, "cost": 0.01},
            },
            "assignments": {
                "ecologic": {"a": "cheap", "b": "strong"},
                "oracle": {"a": "cheap", "b": "cheap"},
            },
        },
        "arrivals": {
            "process": "constant",
            "start_s": 0.0,
            "interarrival_s": 10.0,
            "rate_per_s": 0.1,
            "timestamps_s": [0.0, 10.0],
            "on_off": {
                "lambda_on_per_s": 2.0,
                "lambda_off_per_s": 0.0,
                "mean_on_s": 5.0,
                "mean_off_s": 5.0,
                "initial_state": "ON",
            },
        },
        "router": {"max_concurrency": 8},
        "serverless": {
            "max_instances": 1,
            "max_concurrency_per_instance": 1,
            "idle_timeout_s": 100.0,
            "cold_start_delay_s": 0.0,
            "first_request_cold": False,
            "scale_to_zero": True,
            "queue_discipline": "fifo",
        },
        "sla": {"max_e2e_s": 100.0},
        "policy_order": ["always_cheap"],
        "policies": {
            "always_cheap": {
                "type": "always_cheap",
                "router_overhead_s": 0.0,
                "router_cost_usd": 0.0,
            },
            "always_strong": {
                "type": "always_strong",
                "router_overhead_s": 0.0,
                "router_cost_usd": 0.0,
            },
            "ecologic": {
                "type": "ecologic",
                "router_overhead_s": 0.01,
                "router_cost_usd": 0.0,
            },
            "cost_matched_static": {
                "type": "cost_matched_static",
                "router_overhead_s": 0.0,
                "router_cost_usd": 0.0,
                "f_strong": 0.0,
            },
            "oracle": {
                "type": "oracle",
                "router_overhead_s": 0.0,
                "router_cost_usd": 0.0,
            },
        },
        "sweep": {
            "mode": "ofat",
            "burstiness_sets_process": "on_off",
            "factors": {"arrivals.interarrival_s": [10.0, 20.0]},
        },
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def _sim(cfg, policy="always_cheap", arrivals=None, queries=None):
    cat = catalog_synthetic(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    return simulate(cfg, cat, policy, rng=rng, arrivals_s=arrivals, query_seq=queries)


class TestArrivals(unittest.TestCase):
    def test_constant(self):
        ts = constant_arrivals(4, 2.5, 1.0)
        np.testing.assert_allclose(ts, [1.0, 3.5, 6.0, 8.5])

    def test_trace_replay_exact(self):
        ts = trace_replay([0.0, 5.0, 5.0, 9.0], n=3)
        np.testing.assert_allclose(ts, [0.0, 5.0, 5.0])

    def test_poisson_mean_rate(self):
        rng = np.random.default_rng(0)
        ts = poisson_arrivals(4000, 2.0, rng, 0.0)
        ia = np.diff(np.concatenate([[0.0], ts]))
        self.assertAlmostEqual(float(ia.mean()), 0.5, delta=0.05)

    def test_on_off_zero_off_rate_never_arrives_while_held_off_long(self):
        rng = np.random.default_rng(0)
        ts = on_off_arrivals(
            20,
            lambda_on_per_s=50.0,
            lambda_off_per_s=0.0,
            mean_on_s=1.0,
            mean_off_s=1.0,
            initial_state="ON",
            rng=rng,
            start_s=0.0,
        )
        self.assertEqual(len(ts), 20)
        self.assertTrue(np.all(np.diff(ts) >= -1e-12))

    def test_from_config_requires_process_keys(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(KeyError):
            arrivals_from_config({"process": "poisson", "start_s": 0.0}, 3, rng)


class TestConfig(unittest.TestCase):
    def test_missing_idle_timeout_fails(self):
        cfg = _mini_cfg()
        del cfg["serverless"]["idle_timeout_s"]
        with self.assertRaises(KeyError):
            validate_config(cfg)

    def test_simulated_false_rejected(self):
        cfg = _mini_cfg()
        cfg["simulated"] = False
        with self.assertRaises(ValueError):
            validate_config(cfg)


class TestSimulatorCore(unittest.TestCase):
    def test_sparse_constant_no_queue(self):
        cfg = _mini_cfg()
        out = _sim(cfg, arrivals=np.array([0.0, 10.0]), queries=["a", "a"])
        self.assertTrue(out["SIMULATED"])
        self.assertEqual(out["label"], "SIMULATED")
        self.assertIn("not a measured", out["disclaimer"].lower())
        m = out["metrics"]
        self.assertEqual(m["cold_start_count"], 0)
        self.assertAlmostEqual(m["queue_delay_s"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(m["end_to_end_latency_s"]["mean"], 1.0, places=9)
        self.assertAlmostEqual(m["total_realized_inference_cost"], 0.002)
        self.assertAlmostEqual(m["cost_per_request"], 0.001)
        self.assertAlmostEqual(m["sla_violation_rate"], 0.0)

    def test_backlog_queue_delay(self):
        cfg = _mini_cfg()
        cfg["arrivals"]["interarrival_s"] = 0.5
        out = _sim(cfg, arrivals=np.array([0.0, 0.5]), queries=["a", "a"])
        m = out["metrics"]
        # first 0→1; second waits 0.5, service 1, e2e=1.5; mean e2e=1.25, mean wait=0.25
        self.assertAlmostEqual(m["queue_delay_s"]["mean"], 0.25, places=9)
        self.assertAlmostEqual(m["end_to_end_latency_s"]["mean"], 1.25, places=9)

    def test_cold_start_added_to_e2e(self):
        cfg = _mini_cfg()
        cfg["n_requests"] = 1
        cfg["serverless"]["first_request_cold"] = True
        cfg["serverless"]["cold_start_delay_s"] = 2.0
        out = _sim(cfg, arrivals=np.array([0.0]), queries=["a"])
        m = out["metrics"]
        self.assertEqual(m["cold_start_count"], 1)
        self.assertAlmostEqual(m["cold_start_rate"], 1.0)
        self.assertAlmostEqual(m["end_to_end_latency_s"]["mean"], 3.0, places=9)

    def test_idle_timeout_second_request_is_cold(self):
        cfg = _mini_cfg()
        cfg["serverless"]["first_request_cold"] = True
        cfg["serverless"]["cold_start_delay_s"] = 2.0
        cfg["serverless"]["idle_timeout_s"] = 1.0
        cfg["serverless"]["scale_to_zero"] = True
        out = _sim(cfg, arrivals=np.array([0.0, 20.0]), queries=["a", "a"])
        self.assertEqual(out["metrics"]["cold_start_count"], 2)
        e2e = [r["end_to_end_s"] for r in out["per_request"]]
        self.assertAlmostEqual(e2e[0], 3.0, places=9)
        self.assertAlmostEqual(e2e[1], 3.0, places=9)

    def test_scale_out_instance_is_cold_even_if_first_was_warm(self):
        cfg = _mini_cfg()
        cfg["serverless"]["max_instances"] = 2
        cfg["serverless"]["first_request_cold"] = False
        cfg["serverless"]["cold_start_delay_s"] = 5.0
        cfg["serverless"]["max_concurrency_per_instance"] = 1
        out = _sim(cfg, arrivals=np.array([0.0, 0.0]), queries=["a", "a"])
        cold = [r["cold_start"] for r in out["per_request"]]
        self.assertEqual(sum(cold), 1)
        e2e = sorted(r["end_to_end_s"] for r in out["per_request"])
        self.assertAlmostEqual(e2e[0], 1.0, places=9)
        self.assertAlmostEqual(e2e[1], 6.0, places=9)

    def test_concurrency_two_avoids_queue(self):
        cfg = _mini_cfg()
        cfg["serverless"]["max_concurrency_per_instance"] = 2
        out = _sim(cfg, arrivals=np.array([0.0, 0.0]), queries=["a", "a"])
        self.assertAlmostEqual(out["metrics"]["queue_delay_s"]["mean"], 0.0, places=9)
        self.assertEqual(out["metrics"]["cold_start_count"], 0)

    def test_router_overhead_serial(self):
        cfg = _mini_cfg()
        cfg["router"]["max_concurrency"] = 1
        cfg["policies"]["always_cheap"]["router_overhead_s"] = 0.5
        cfg["n_requests"] = 1
        out = _sim(cfg, arrivals=np.array([0.0]), queries=["a"])
        self.assertAlmostEqual(out["metrics"]["end_to_end_latency_s"]["mean"], 1.5, places=9)
        self.assertAlmostEqual(out["metrics"]["mean_router_overhead_s"], 0.5)

    def test_sla_all_violate(self):
        cfg = _mini_cfg()
        cfg["sla"]["max_e2e_s"] = 0.5
        out = _sim(cfg, arrivals=np.array([0.0, 10.0]), queries=["a", "a"])
        self.assertAlmostEqual(out["metrics"]["sla_violation_rate"], 1.0)

    def test_always_strong_cost(self):
        cfg = _mini_cfg()
        cfg["policy_order"] = ["always_strong"]
        out = _sim(cfg, "always_strong", arrivals=np.array([0.0, 10.0]), queries=["a", "a"])
        self.assertAlmostEqual(out["metrics"]["total_realized_inference_cost"], 0.02)
        self.assertAlmostEqual(out["metrics"]["end_to_end_latency_s"]["mean"], 4.0, places=9)

    def test_cost_matched_f_one_is_all_strong(self):
        cfg = _mini_cfg()
        cfg["policies"]["cost_matched_static"]["f_strong"] = 1.0
        cfg["policy_order"] = ["cost_matched_static"]
        out = _sim(
            cfg, "cost_matched_static", arrivals=np.array([0.0, 10.0]), queries=["a", "b"]
        )
        self.assertTrue(all(r["model"] == "strong" for r in out["per_request"]))

    def test_oracle_uses_assignment_map(self):
        cfg = _mini_cfg()
        cfg["policy_order"] = ["oracle"]
        out = _sim(cfg, "oracle", arrivals=np.array([0.0, 10.0]), queries=["a", "b"])
        self.assertEqual([r["model"] for r in out["per_request"]], ["cheap", "cheap"])

    def test_ecologic_router_overhead_in_e2e(self):
        cfg = _mini_cfg()
        cfg["policy_order"] = ["ecologic"]
        out = _sim(cfg, "ecologic", arrivals=np.array([0.0, 20.0]), queries=["a", "b"])
        rows = {r["query_id"]: r for r in out["per_request"]}
        self.assertAlmostEqual(rows["a"]["end_to_end_s"], 1.01, places=9)
        self.assertAlmostEqual(rows["b"]["end_to_end_s"], 4.01, places=9)

    def test_p95_p99_present(self):
        cfg = _mini_cfg()
        out = _sim(cfg, arrivals=np.array([0.0, 10.0]), queries=["a", "a"])
        self.assertIsNotNone(out["metrics"]["p50_latency_s"])
        self.assertIsNotNone(out["metrics"]["p95_latency_s"])
        self.assertIsNotNone(out["metrics"]["p99_latency_s"])
        self.assertGreater(out["metrics"]["throughput_per_s"], 0.0)


class TestSweep(unittest.TestCase):
    def test_ofat_adds_non_baseline_levels(self):
        cfg = _mini_cfg()
        snaps = expand_ofat(cfg)
        factors = [s[0] for s in snaps]
        self.assertIn("baseline", factors)
        self.assertIn("arrivals.interarrival_s", factors)

    def test_run_sweeps_labels_simulated(self):
        cfg = _mini_cfg()
        cfg["n_requests"] = 2
        payload = run_sweeps(cfg)
        self.assertTrue(payload["SIMULATED"])
        self.assertGreaterEqual(payload["n_runs"], 2)
        for run in payload["runs"]:
            self.assertTrue(run["SIMULATED"])
            if run.get("skipped"):
                continue
            self.assertIn("metrics", run)
            self.assertTrue(run["metrics"]["SIMULATED"])


if __name__ == "__main__":
    unittest.main()
