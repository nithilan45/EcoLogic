"""Systems stress-test suite: MEASURED vs SIMULATED isolation, workloads, caps."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from woais_experiments.paths import RESULTS, reset_run_context, set_run_context
from woais_experiments.stress import MEASURED, SIMULATED
from woais_experiments.stress.analyze_stress import analyze_cells, cell_metrics, pareto_under_load
from woais_experiments.stress.failure_analysis import classify_row, summarize_failures
from woais_experiments.stress.load_generator import CostCapExceeded, CostGuard, run_open_loop
from woais_experiments.stress.records import MixedSourceError, assert_homogeneous, occupancy_at_arrivals
from woais_experiments.stress.saturation import detect_saturation, occupancy_growth
from woais_experiments.stress.stress_test import load_stress_config, main as stress_main, run_suite
from woais_experiments.stress.workload_profiles import (
    LOAD_MULTIPLIERS,
    PROFILE_NAMES,
    arrivals_for_profile,
    profile_seed,
    safe_multipliers,
)

SLA = {
    "p95_latency_s": 1.5,
    "failure_rate": 0.05,
    "queue_growth_min_slope": 0.05,
    "queue_growth_min_r2": 0.30,
}


def _tmp_root() -> Path:
    root = RESULTS / "_stress_unittest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


class TestWorkloadProfiles(unittest.TestCase):
    def test_all_ten_profiles_emit_n_arrivals(self):
        replay = [float(i) * 0.2 for i in range(20)]
        for name in PROFILE_NAMES:
            spec = arrivals_for_profile(name, 10, multiplier=1, seed=20260909, timestamps_s=replay)
            self.assertEqual(len(spec["arrivals_s"]), 10)
            self.assertTrue(np.all(np.diff(spec["arrivals_s"]) >= -1e-12))

    def test_multiplier_compresses_constant_horizon(self):
        a = arrivals_for_profile("steady_medium", 9, multiplier=1, seed=1)
        b = arrivals_for_profile("steady_medium", 9, multiplier=2, seed=1)
        h1 = float(a["arrivals_s"][-1] - a["arrivals_s"][0])
        h2 = float(b["arrivals_s"][-1] - b["arrivals_s"][0])
        self.assertAlmostEqual(h1 / h2, 2.0, places=6)

    def test_replay_time_compression(self):
        ts = [0.0, 2.0, 4.0, 6.0]
        a = arrivals_for_profile("replay", 4, multiplier=1, timestamps_s=ts, seed=1)
        b = arrivals_for_profile("replay", 4, multiplier=2, timestamps_s=ts, seed=1)
        self.assertAlmostEqual(float(b["horizon_s"]) * 2.0, float(a["horizon_s"]), places=6)

    def test_poisson_is_deterministic_given_seed(self):
        a = arrivals_for_profile("poisson", 12, multiplier=3, seed=20260909)
        b = arrivals_for_profile("poisson", 12, multiplier=3, seed=20260909)
        np.testing.assert_array_equal(a["arrivals_s"], b["arrivals_s"])
        c = arrivals_for_profile("poisson", 12, multiplier=3, seed=7)
        self.assertFalse(np.allclose(a["arrivals_s"], c["arrivals_s"]))

    def test_profile_seed_ignores_python_hash_salt(self):
        self.assertEqual(profile_seed(1, "poisson", 2), profile_seed(1, "poisson", 2))
        self.assertNotEqual(profile_seed(1, "poisson", 2), profile_seed(1, "poisson", 5))

    def test_safe_multipliers_drop_paid_high_load(self):
        paid = safe_multipliers(mode="measured", paid=True, requested=LOAD_MULTIPLIERS)
        self.assertEqual(paid, [1.0, 2.0])
        stub = safe_multipliers(mode="measured", paid=False, requested=LOAD_MULTIPLIERS)
        self.assertEqual(stub, [1.0, 2.0, 5.0, 10.0])
        sim = safe_multipliers(mode="simulated", paid=False, requested=LOAD_MULTIPLIERS)
        self.assertEqual(sim, [1.0, 2.0, 5.0, 10.0, 25.0, 50.0])


class TestOccupancyAndSaturation(unittest.TestCase):
    def test_occupancy_counts_in_system_before_arrival(self):
        occ = occupancy_at_arrivals([0.0, 1.0, 2.0], [2.5, 2.6, 2.7])
        self.assertEqual(occ, [0, 1, 2])

    def test_saturation_p95_and_queue_growth(self):
        rows = []
        for i in range(8):
            rows.append({
                "measurement_type": SIMULATED,
                "request_id": str(i),
                "policy": "always_cheap",
                "stress_profile": "steady_high",
                "load_multiplier": 10,
                "scheduled_arrival_s": float(i),
                "complete_s": 50.0 + i,
                "end_to_end_s": 50.0,
                "queue_wait_s": 49.0,
                "http_status": 200,
                "error_type": None,
                "retry_count": 0,
            })
        sat = detect_saturation(rows, sla_p95_s=1.0, failure_rate_threshold=0.05)
        self.assertTrue(sat["saturated"])
        self.assertIn("p95_exceeds_sla", sat["reasons"])
        self.assertTrue(sat["environment_only"])
        growth = occupancy_growth([r["scheduled_arrival_s"] for r in rows], list(range(8)))
        self.assertTrue(growth["persistent_growth"])

    def test_not_saturated_when_fast_and_stable(self):
        rows = []
        for i in range(8):
            rows.append({
                "measurement_type": SIMULATED,
                "request_id": str(i),
                "policy": "always_cheap",
                "stress_profile": "steady_low",
                "load_multiplier": 1,
                "scheduled_arrival_s": float(i),
                "complete_s": float(i) + 0.05,
                "end_to_end_s": 0.05,
                "http_status": 200,
                "error_type": None,
                "retry_count": 0,
            })
        sat = detect_saturation(rows, sla_p95_s=1.0, failure_rate_threshold=0.05)
        self.assertFalse(sat["saturated"])


class TestFailureTaxonomy(unittest.TestCase):
    def test_classifies_timeout_429_5xx_and_cost_cap(self):
        self.assertEqual(classify_row({"error_type": "timeout", "http_status": 0}), "timeout")
        self.assertEqual(classify_row({"http_status": 429}), "http_429")
        self.assertEqual(classify_row({"http_status": 503, "error_type": "http_5xx"}), "http_5xx")
        self.assertEqual(classify_row({"error_type": "cost_cap_stop"}), "cost_cap_stop")
        self.assertEqual(classify_row({"http_status": 200}), "ok")
        summary = summarize_failures([
            {"http_status": 200, "retry_count": 1},
            {"error_type": "timeout", "http_status": 0, "retry_count": 2},
        ])
        self.assertAlmostEqual(summary["timeout_rate"], 0.5)
        self.assertAlmostEqual(summary["retry_rate"], 1.0)


class TestCostCap(unittest.TestCase):
    def test_paid_requires_limit(self):
        with self.assertRaises(ValueError):
            CostGuard(None, estimate_usd_per_query=0.01, paid=True)

    def test_open_loop_stops_when_realized_exceeds_cap(self):
        guard = CostGuard(2.5, estimate_usd_per_query=0.01, paid=True)
        arrivals = [0.0, 0.01, 0.02, 0.03, 0.04]

        def submit(i, payload):
            return {
                "measurement_type": MEASURED,
                "request_id": str(i),
                "realized_provider_cost": 1.0,
                "http_status": 200,
                "error_type": None,
                "retry_count": 0,
                "end_to_end_ms": 1.0,
                "provider_request_ms": 1.0,
            }

        payloads = [
            {
                "request_id": str(i),
                "policy": "always_cheap",
                "canonical_policy": "always_cheap",
                "stress_profile": "steady_low",
                "load_multiplier": 1,
            }
            for i in range(5)
        ]
        out = run_open_loop(arrivals, payloads, submit, max_workers=1, cost_guard=guard)
        self.assertTrue(out["stopped_on_cost_cap"])
        self.assertLess(out["n_issued"], 5)
        self.assertGreater(guard.realized, 0.0)
        self.assertTrue(any(r.get("cost_cap_stopped") or r.get("error_type") == "cost_cap_stop" for r in out["records"]))
        for row in out["records"]:
            self.assertEqual(row["measurement_type"], MEASURED)
            self.assertNotIn("des_wait_s", row)

    def test_single_request_realized_over_cap(self):
        guard = CostGuard(0.5, estimate_usd_per_query=0.01, paid=True)

        def submit(i, payload):
            return {
                "measurement_type": MEASURED,
                "request_id": str(i),
                "realized_provider_cost": 2.0,
                "http_status": 200,
                "error_type": None,
                "retry_count": 0,
                "end_to_end_ms": 1.0,
            }

        out = run_open_loop(
            [0.0],
            [{"request_id": "0", "canonical_policy": "always_cheap", "stress_profile": "steady_low", "load_multiplier": 1}],
            submit,
            max_workers=1,
            cost_guard=guard,
        )
        self.assertTrue(out["stopped_on_cost_cap"])
        self.assertGreater(guard.realized, 0.5)


class TestIsolation(unittest.TestCase):
    def test_analyzer_refuses_mixed_records(self):
        rows = [
            {
                "measurement_type": MEASURED,
                "request_id": "a",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1,
                "scheduled_arrival_s": 0.0,
                "complete_s": 0.1,
                "end_to_end_s": 0.1,
                "http_status": 200,
                "timestamp": "t",
                "query_length_chars": 1,
                "input_tokens": 1,
                "router_decision_ms": 0.1,
                "provider_request_ms": 1.0,
                "time_to_first_token_ms": 1.0,
                "generation_ms": 1.0,
                "end_to_end_ms": 100.0,
                "selected_model": "cheap",
                "output_tokens": 1,
                "realized_provider_cost": 0.001,
                "error_type": None,
                "retry_count": 0,
                "suite": "stress_measured",
            },
            {
                "measurement_type": SIMULATED,
                "request_id": "b",
                "policy": "ecologic",
                "stress_profile": "poisson",
                "load_multiplier": 1,
                "scheduled_arrival_s": 0.0,
                "des_wait_s": 0.01,
                "simulator": True,
            },
        ]
        with self.assertRaises(MixedSourceError):
            assert_homogeneous(rows, MEASURED)

    def test_measured_record_cannot_carry_simulator_keys(self):
        row = {
            "measurement_type": MEASURED,
            "request_id": "a",
            "policy": "ecologic",
            "stress_profile": "poisson",
            "load_multiplier": 1,
            "scheduled_arrival_s": 0.0,
            "timestamp": "t",
            "query_length_chars": 1,
            "input_tokens": 1,
            "router_decision_ms": 0.1,
            "provider_request_ms": 1.0,
            "time_to_first_token_ms": 1.0,
            "generation_ms": 1.0,
            "end_to_end_ms": 100.0,
            "selected_model": "cheap",
            "output_tokens": 1,
            "realized_provider_cost": 0.001,
            "http_status": 200,
            "error_type": None,
            "retry_count": 0,
            "des_wait_s": 0.2,
        }
        with self.assertRaises(MixedSourceError):
            assert_homogeneous([row], MEASURED)


class TestParetoUnderLoad(unittest.TestCase):
    def test_ecologic_undominated_when_better_quality_at_mid_cost(self):
        cells = []
        for name, cost, qual in (
            ("always_cheap", 0.001, 0.70),
            ("always_strong", 0.020, 0.93),
            ("ecologic", 0.008, 0.90),
            ("cost_matched_static", 0.008, 0.80),
        ):
            cells.append({
                "profile": "poisson",
                "multiplier": 2.0,
                "policy": name,
                "cost_per_query": cost,
                "quality_mean": qual,
                "p95_latency_s": 0.2,
            })
        report = pareto_under_load(cells)[0]
        self.assertTrue(report["ecologic_pareto_efficient"])
        self.assertIn("ecologic", report["undominated"])


class TestSimulatedSuite(unittest.TestCase):
    def setUp(self):
        self.root = _tmp_root()
        self.tokens = set_run_context(output_root=self.root, overwrite_policy="replace")

    def tearDown(self):
        reset_run_context(self.tokens)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_simulated_run_tags_and_resume(self):
        cfg = load_stress_config()
        result = run_suite(
            cfg,
            mode="simulated",
            allow_api=False,
            dry_run=True,
            max_cost_usd=None,
            resume=False,
            profiles=["poisson", "sudden_spike"],
            multipliers=[1, 5],
            n_requests=8,
            max_multiplier=5,
        )
        self.assertEqual(result["measurement_type"], SIMULATED)
        self.assertGreaterEqual(result["n_cells"], 8)
        analysis = result["analysis"]
        self.assertEqual(analysis["measurement_type"], SIMULATED)
        self.assertTrue(analysis["environment_only"])
        req_dir = self.root / "stress" / "simulated" / "requests"
        files = list(req_dir.glob("*.jsonl"))
        self.assertTrue(files)
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertEqual(row["measurement_type"], SIMULATED)
                self.assertEqual(row["suite"], "stress_simulated")
        state_path = self.root / "stress" / "simulated" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        n_complete = sum(1 for c in state["cells"].values() if c["status"] == "complete")
        resumed = run_suite(
            cfg,
            mode="simulated",
            allow_api=False,
            dry_run=True,
            max_cost_usd=None,
            resume=True,
            profiles=["poisson", "sudden_spike"],
            multipliers=[1, 5],
            n_requests=8,
            max_multiplier=5,
        )
        self.assertEqual(resumed["n_cells"], n_complete)
        self.assertIn("max_sustainable_throughput", resumed["analysis"])
        self.assertTrue(resumed["analysis"]["max_sustainable_throughput"]["environment_only"])

    def test_simulated_cost_cap_stops_suite(self):
        cfg = load_stress_config()
        result = run_suite(
            cfg,
            mode="simulated",
            allow_api=False,
            dry_run=True,
            max_cost_usd=0.0001,
            resume=False,
            profiles=["steady_high"],
            multipliers=[1, 2],
            n_requests=8,
            max_multiplier=2,
        )
        self.assertTrue(result["stopped_on_cost_cap"])
        self.assertLess(result["n_cells"], 8)

    def test_cli_refuses_allow_api_on_simulated(self):
        code = stress_main(["--mode", "simulated", "--allow-api", "--max-cost-usd", "1"])
        self.assertEqual(code, 2)

    def test_cli_refuses_paid_without_max_cost(self):
        code = stress_main(["--mode", "measured", "--allow-api"])
        self.assertEqual(code, 2)


class TestMeasuredDryRun(unittest.TestCase):
    def setUp(self):
        self.root = _tmp_root()
        self.tokens = set_run_context(output_root=self.root, overwrite_policy="replace")

    def tearDown(self):
        reset_run_context(self.tokens)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_measured_stub_is_tagged_measured(self):
        cfg = load_stress_config()
        result = run_suite(
            cfg,
            mode="measured",
            allow_api=False,
            dry_run=True,
            max_cost_usd=None,
            resume=False,
            profiles=["steady_high"],
            multipliers=[1],
            n_requests=4,
            max_multiplier=1,
        )
        self.assertEqual(result["measurement_type"], MEASURED)
        self.assertEqual(result["n_cells"], 4)
        req_dir = self.root / "stress" / "measured" / "requests"
        for path in req_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertEqual(row["measurement_type"], MEASURED)
                self.assertNotIn("des_wait_s", row)
                self.assertNotIn("simulator", row)
                metrics = cell_metrics(
                    [row],
                    measurement_type=MEASURED,
                    sla=SLA,
                    capacity=8,
                    strong_model=None,
                    profile="steady_high",
                    multiplier=1,
                    policy=row["policy"],
                )
                self.assertEqual(metrics["measurement_type"], MEASURED)
        analysis = analyze_cells(result["analysis"]["cells"], measurement_type=MEASURED)
        self.assertTrue(analysis["environment_only"])


if __name__ == "__main__":
    unittest.main()
