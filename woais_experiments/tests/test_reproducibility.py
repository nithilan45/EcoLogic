"""Reproducibility / anonymity validator. Does not mutate frozen experiment tables."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.paths import RESULTS, ROOT as REPO
from woais_experiments.reproducibility.anonymity_scan import load_anonymity_config, scan_tree
from woais_experiments.reproducibility.artifact_manifest import package_check
from woais_experiments.reproducibility.clean_clone_test import copy_clean_tree
from woais_experiments.reproducibility.environment_capture import capture_environment
from woais_experiments.reproducibility.reproduce_tables import reconstruct, smoke_external_public
from woais_experiments.reproducibility.secret_scan import scan_file, scan_tree as scan_secrets
from woais_experiments.runner.cli import main
from woais_experiments.runner.session import COMMANDS


class TestEnvironmentCapture(unittest.TestCase):
    def test_records_python_git_and_seeds(self):
        env = capture_environment()
        self.assertRegex(env["python"]["version"], r"^3\.\d+")
        self.assertIn("random_seeds", env)
        self.assertEqual(len(env["config_hash"]), 64)
        self.assertTrue(env["artifact_hashes"])
        self.assertIsInstance(env["cwd_is_repo"], bool)
        self.assertNotIn("OPENAI_API_KEY", json.dumps(env.get("paid_api_key_names_present")))


class TestSecretScan(unittest.TestCase):
    def test_reports_type_not_value(self):
        tmp = Path(tempfile.mkdtemp())
        token = "ghp_" + ("B" * 40)
        (tmp / "leak.txt").write_text(f"export TOKEN={token}\n", encoding="utf-8")
        hits = scan_file(tmp / "leak.txt")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["secret_type"], "github_pat")
        dumped = json.dumps(hits)
        self.assertNotIn(token, dumped)
        self.assertNotIn("B" * 40, dumped)

    def test_dotenv_without_values(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / ".env").write_text("OPENAI_API_KEY=should-not-appear\n", encoding="utf-8")
        hits = scan_file(tmp / ".env")
        self.assertEqual(hits[0]["secret_type"], "dotenv_file")
        self.assertNotIn("should-not-appear", json.dumps(hits))

    def test_env_lookup_is_not_a_secret(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "app.py").write_text(
            'TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")\n'
            'TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"\n',
            encoding="utf-8",
        )
        self.assertEqual(scan_file(tmp / "app.py"), [])


class TestAnonymityScan(unittest.TestCase):
    def test_config_loads(self):
        cfg = load_anonymity_config()
        self.assertIn("identities", cfg)
        self.assertEqual(cfg["severity"], "critical")

    def test_flags_configured_identity_in_temp_tree(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "note.md").write_text("Contact Zelda Reviewer for details.\n", encoding="utf-8")
        cfg = {
            "severity": "critical",
            "scan_git_metadata": False,
            "identities": {"names": ["Zelda Reviewer"]},
            "path_patterns": [],
            "exclude_globs": [],
            "exclude_suffixes": [".png"],
        }
        report = scan_tree(tmp, config=cfg)
        self.assertFalse(report["ok"])
        self.assertIn("note.md", report["files_requiring_review"])
        dumped = json.dumps(report)
        self.assertNotIn("Zelda Reviewer", dumped)


class TestCleanCopy(unittest.TestCase):
    def test_excludes_venv_and_env(self):
        src = Path(tempfile.mkdtemp())
        (src / "keep.py").write_text("x = 1\n", encoding="utf-8")
        (src / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (src / ".venv").mkdir()
        (src / ".venv" / "pyvenv.cfg").write_text("home = /tmp\n", encoding="utf-8")
        (src / "__pycache__").mkdir()
        (src / "__pycache__" / "keep.cpython-313.pyc").write_bytes(b"\x00")
        dest = Path(tempfile.mkdtemp()) / "repo"
        copy_clean_tree(dest, src=src)
        self.assertTrue((dest / "keep.py").exists())
        self.assertFalse((dest / ".env").exists())
        self.assertFalse((dest / ".venv").exists())
        self.assertFalse((dest / "__pycache__").exists())

    def test_git_metadata_dir_is_not_copied(self):
        from woais_experiments.reproducibility.clean_clone_test import IGNORE_DIR_NAMES, _ignore

        self.assertIn(".git", IGNORE_DIR_NAMES)
        skipped = _ignore("/tmp/repo", [".git", "keep.py", ".venv"])
        self.assertIn(".git", skipped)
        self.assertIn(".venv", skipped)
        self.assertNotIn("keep.py", skipped)


class TestReconstruction(unittest.TestCase):
    def test_headline_tables_trace_to_raw(self):
        out = reconstruct()
        self.assertTrue(out["ok"], msg=out)
        names = {r["result_name"] for r in out["records"]}
        self.assertIn("accounting/stage12.json", names)
        self.assertIn("routing/stage12_policies.json", names)
        self.assertIn("external/routellm_tables.json", names)
        for rec in out["records"]:
            self.assertTrue(rec["traced_to_raw"], msg=rec)
            self.assertTrue(rec["matches_recomputed"], msg=rec)
            self.assertTrue(rec["input_hashes"])
            self.assertIsNotNone(rec["output_hash"])

    def test_missing_aggregate_fails_trace(self):
        from woais_experiments.reproducibility.reproduce_tables import _trace_accounting

        rec = _trace_accounting(Path("/tmp/woais_missing_stage12.json"))
        self.assertFalse(rec["traced_to_raw"])
        self.assertIsNone(rec["matches_recomputed"])

    def test_external_public_smoke(self):
        smoke = smoke_external_public()
        self.assertTrue(smoke["ok"])
        self.assertGreater(smoke["n_items"], 0)


class TestPackageCheck(unittest.TestCase):
    def test_lockfile_present(self):
        report = package_check(REPO)
        self.assertTrue(report["ok"] or not any(
            c.get("reason") == "missing_lockfile" for c in report["critical"]
        ))
        self.assertTrue((REPO / "woais_experiments" / "requirements-lock.txt").exists())


class TestValidateCommand(unittest.TestCase):
    def test_command_is_registered(self):
        self.assertIn("validate-artifact", COMMANDS)

    def test_skip_clone_writes_reports_without_touching_frozen(self):
        graded = REPO / "raw_results" / "graded.jsonl"
        before = graded.stat().st_mtime
        code = main(["validate-artifact", "--skip-clone"])
        self.assertEqual(code, 1)
        self.assertEqual(graded.stat().st_mtime, before)
        dest = RESULTS / "reproducibility"
        self.assertTrue((dest / "result_provenance.json").exists())
        self.assertTrue((dest / "ANONYMITY_REPORT.md").exists())
        self.assertTrue((dest / "secret_scan.json").exists())
        self.assertTrue((dest / "environment.json").exists())
        self.assertTrue((dest / "validate.json").exists())
        provenance = json.loads((dest / "result_provenance.json").read_text())
        self.assertIn("records", provenance)
        secrets = json.loads((dest / "secret_scan.json").read_text())
        for hit in secrets.get("findings") or []:
            self.assertEqual(set(hit.keys()), {"file", "line", "secret_type"})
        report = json.loads((dest / "validate.json").read_text())
        self.assertTrue(report["clean_clone"]["skipped"])
        self.assertFalse(report["clean_clone"]["ok"])
        self.assertIn("clean_clone_skipped", report["critical"])
        md = (dest / "ANONYMITY_REPORT.md").read_text()
        self.assertIn("Files requiring manual review", md)


if __name__ == "__main__":
    unittest.main()
