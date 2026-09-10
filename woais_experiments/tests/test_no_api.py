import inspect
import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.paths import ROOT as REPO
from woais_experiments import run_offline


class TestNoApiCalls(unittest.TestCase):
    def test_run_offline_source_never_calls_chat(self):
        src = inspect.getsource(run_offline)
        self.assertNotIn("chat(", src)
        self.assertNotIn("TOGETHER_API_KEY", src)
        self.assertNotIn("OPENAI_API_KEY", src)
        self.assertNotIn("httpx", src)

    def test_package_does_not_import_generation_runners(self):
        src = inspect.getsource(run_offline)
        for banned in ("run_benchmark", "s7_run", "run_pool", "quality_benchmark_harness"):
            self.assertNotIn(banned, src)

    def test_hash_manifest_lives_in_package(self):
        self.assertTrue((REPO / "woais_experiments" / "EXISTING_RESULTS_SHA256.txt").exists())


if __name__ == "__main__":
    unittest.main()
