import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.accounting.costs import usd_from_tokens, usd_rates_per_million
from woais_experiments.paths import ensure_legacy_imports


class TestAccounting(unittest.TestCase):
    def test_usage_and_cost_matches_published_rates(self):
        usd = usd_from_tokens("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        # 1000/1e6 * 2.50 + 500/1e6 * 10.00
        self.assertAlmostEqual(usd, 0.0025 + 0.005, places=12)

    def test_qwen_rate_table_imported_from_api(self):
        rates = usd_rates_per_million()
        self.assertEqual(rates["Qwen/Qwen3.5-9B"]["input"], 0.17)
        self.assertEqual(rates["openai/gpt-oss-20b"]["output"], 0.20)

    def test_usage_and_cost_function_is_the_legacy_one(self):
        ensure_legacy_imports()
        try:
            from api import usage_and_cost
        except Exception as exc:
            self.skipTest(f"benchmark.api not importable: {exc}")
        payload = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"finish_reason": "stop"}],
        }
        got = usage_and_cost("openai/gpt-oss-20b", payload)
        self.assertEqual(got["prompt_tokens"], 10)
        self.assertEqual(got["completion_tokens"], 20)
        self.assertAlmostEqual(got["usd"], 10 / 1e6 * 0.05 + 20 / 1e6 * 0.20)


if __name__ == "__main__":
    unittest.main()
