import unittest

import numpy as np

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.latency.serverless import fit_linear_service, flag_cold_like, simulate_fcfs


class TestServerlessModel(unittest.TestCase):
    def test_fcfs_no_queue_when_arrivals_are_sparse(self):
        service = np.array([1.0, 1.0, 1.0])
        arrivals = np.array([0.0, 10.0, 20.0])
        out = simulate_fcfs(service, arrivals, n_servers=1, idle_timeout_s=None)
        self.assertAlmostEqual(out["mean_wait_s"], 0.0, places=9)
        self.assertAlmostEqual(out["mean_sojourn_s"], 1.0, places=9)
        self.assertEqual(out["cold_frac"], 0.0)

    def test_fcfs_queues_under_backlog(self):
        service = np.array([5.0, 5.0])
        arrivals = np.array([0.0, 1.0])
        out = simulate_fcfs(service, arrivals, n_servers=1)
        # first: wait 0 sojourn 5; second: start at 5, wait 4, sojourn 9
        self.assertAlmostEqual(out["mean_wait_s"], 2.0, places=9)
        self.assertAlmostEqual(out["mean_sojourn_s"], 7.0, places=9)

    def test_idle_timeout_adds_cold_penalty(self):
        service = np.array([1.0, 1.0])
        arrivals = np.array([0.0, 20.0])
        out = simulate_fcfs(
            service, arrivals, n_servers=1,
            idle_timeout_s=5.0, cold_penalty_s=3.0, first_request_cold=True,
        )
        self.assertEqual(out["cold_frac"], 1.0)
        # first: 3 cold + 1 service = 4; second: idle 20-(0+4)=16 > 5, +3 cold +1 = 4
        self.assertAlmostEqual(out["mean_sojourn_s"], 4.0, places=9)
        self.assertAlmostEqual(out["mean_wait_s"], 0.0, places=9)
        self.assertAlmostEqual(out["mean_queue_wait_s"], 0.0, places=9)
        self.assertAlmostEqual(out["mean_cold_s"], 3.0, places=9)
        # utilization includes cold busy time: busy=(1+3)*2, horizon=24
        self.assertAlmostEqual(out["utilization"], 8.0 / 24.0, places=9)

    def test_two_servers_avoid_the_backlog(self):
        service = np.array([5.0, 5.0])
        arrivals = np.array([0.0, 1.0])
        out = simulate_fcfs(service, arrivals, n_servers=2)
        self.assertAlmostEqual(out["mean_wait_s"], 0.0, places=9)

    def test_linear_fit_recovers_slope(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x
        fit = fit_linear_service(y, x)
        self.assertAlmostEqual(fit["alpha"], 2.0, places=6)
        self.assertAlmostEqual(fit["beta"], 0.5, places=6)
        self.assertAlmostEqual(fit["r2"], 1.0, places=6)
        flags = flag_cold_like(fit["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))


class TestColdLikeResidualTolerance(unittest.TestCase):
    def test_perfectly_linear_timings_have_no_outliers(self):
        x = np.linspace(0, 10, 40)
        y = 1.5 + 0.25 * x
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))

    def test_linear_with_machine_epsilon_is_not_cold(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x
        y = y + np.finfo(float).eps * np.arange(len(y), dtype=float)
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))

    def test_one_genuine_latency_outlier(self):
        x = np.linspace(0, 10, 40)
        y = 2.0 + 0.5 * x
        y = y.copy()
        y[11] += 8.0
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertTrue(bool(flags[11]))
        self.assertEqual(int(flags.sum()), 1)

    def test_multiple_genuine_latency_outliers(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x
        y = y.copy()
        y[5] += 9.0
        y[30] += 9.0
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertTrue(bool(flags[5]))
        self.assertTrue(bool(flags[30]))
        self.assertGreaterEqual(int(flags.sum()), 2)

    def test_constant_latency_is_not_cold(self):
        x = np.linspace(0, 10, 25)
        y = np.full(25, 0.42)
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))
        flags0 = flag_cold_like(np.zeros(12), k=2.0)
        self.assertFalse(bool(flags0.any()))

    def test_extremely_small_values_are_not_cold(self):
        x = np.linspace(0, 1, 30)
        y = 1e-18 + 1e-20 * x
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))

    def test_very_large_linear_values_are_not_cold(self):
        x = np.linspace(0, 10, 50)
        y = 1.0e8 + 0.5 * x
        flags = flag_cold_like(fit_linear_service(y, x)["residuals"], k=2.0)
        self.assertFalse(bool(flags.any()))


if __name__ == "__main__":
    unittest.main()
