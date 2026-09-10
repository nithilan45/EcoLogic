import unittest

from woais_experiments.tests import ROOT  # noqa: F401 — path bootstrap
from woais_experiments.frozen import parse_manifest, sha256_file, verify_frozen_hashes


class TestFrozenHashes(unittest.TestCase):
    def test_manifest_has_expected_count(self):
        recs = parse_manifest()
        self.assertEqual(len(recs), 67)
        paths = {r.relpath for r in recs}
        self.assertIn("raw_results/graded.jsonl", paths)
        self.assertIn("raw_results/analysis_cost.json", paths)
        self.assertIn("stage7_10/s9_static_baselines.json", paths)
        self.assertIn("stage7_10/regret_correction_validation.json", paths)

    def test_every_hashed_file_matches(self):
        check = verify_frozen_hashes()
        self.assertTrue(check["ok"], msg=check)
        self.assertEqual(check["n_mismatches"], 0)
        self.assertEqual(check["n_missing"], 0)

    def test_sha256_helper_agrees_with_manifest_row(self):
        rec = next(r for r in parse_manifest() if r.relpath.endswith("tables_cost.md"))
        self.assertEqual(sha256_file(rec.path), rec.sha256)
        self.assertEqual(rec.path.stat().st_size, rec.size)


if __name__ == "__main__":
    unittest.main()
