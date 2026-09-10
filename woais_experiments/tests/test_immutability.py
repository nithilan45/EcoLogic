import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.frozen import FrozenTreeError, open_frozen, write_result
from woais_experiments.paths import RESULTS, ROOT as REPO, is_frozen


class TestImmutability(unittest.TestCase):
    def test_graded_jsonl_is_frozen(self):
        self.assertTrue(is_frozen(REPO / "raw_results" / "graded.jsonl"))

    def test_open_frozen_rejects_write_modes(self):
        path = REPO / "raw_results" / "tables_cost.md"
        for mode in ("w", "a", "w+", "x"):
            with self.assertRaises(FrozenTreeError):
                open_frozen(path, mode)

    def test_write_result_cannot_escape_results_dir(self):
        with self.assertRaises(RuntimeError):
            write_result(REPO / "raw_results" / "should_not_exist.json", {"no": True})

    def test_write_result_lands_under_results(self):
        dest = write_result("immutability_probe.json", {"ok": True})
        self.assertTrue(str(dest).startswith(str(RESULTS.resolve())))
        self.assertTrue(dest.exists())
        dest.unlink()

    def test_woais_results_are_not_frozen_trees(self):
        self.assertFalse(is_frozen(RESULTS / "summary.json"))


if __name__ == "__main__":
    unittest.main()
