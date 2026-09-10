"""Reproducible experiment runner: config hash, no silent overwrite, no paid APIs."""

from __future__ import annotations

import inspect
import json
import shutil
import unittest

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.frozen import ResultExistsError, write_result
from woais_experiments.paths import RUNS, reset_run_context, set_run_context
from woais_experiments.runner.cli import main
from woais_experiments.runner.config import config_hash, load_run_config, seeds_from_config
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.runner.stages import stages_for_command


class TestConfig(unittest.TestCase):
    def test_yaml_loads_deterministic_seeds(self):
        cfg = load_run_config()
        self.assertFalse(cfg["api_calls"])
        seeds = seeds_from_config(cfg)
        self.assertEqual(seeds["monte_carlo"], 20260909)
        self.assertEqual(seeds["random_policy"], 20260905)

    def test_config_hash_stable(self):
        a = config_hash(load_run_config())
        b = config_hash(load_run_config())
        self.assertEqual(len(a), 64)
        self.assertEqual(a, b)


class TestStageList(unittest.TestCase):
    def test_all_order(self):
        self.assertEqual(
            stages_for_command("all"),
            (
                "audit",
                "accounting",
                "static",
                "oracle",
                "latency",
                "workload",
                "external",
                "robustness",
            ),
        )


class TestNoPaidApi(unittest.TestCase):
    def test_runner_source_has_no_chat(self):
        from woais_experiments.runner import cli, stages, session
        for mod in (cli, stages, session):
            src = inspect.getsource(mod)
            self.assertNotIn("huggingface", src)
            self.assertNotIn("httpx.Client", src)


class TestGitSnapshot(unittest.TestCase):
    def test_commit_is_hex_when_git_is_present(self):
        snap = git_snapshot()
        if not snap["available"]:
            self.skipTest("git metadata is not required for a clean source tree")
        self.assertEqual(len(snap["commit"]), 40)


class TestAuditRun(unittest.TestCase):
    def tearDown(self):
        rid = getattr(self, "run_id", None)
        if rid:
            dest = RUNS / rid
            if dest.exists():
                shutil.rmtree(dest)

    def test_audit_writes_run_dir_and_summary(self):
        self.run_id = "unit_audit_runner"
        dest = RUNS / self.run_id
        if dest.exists():
            shutil.rmtree(dest)
        code = main(["audit", "--run-id", self.run_id, "--no-symlink"])
        self.assertEqual(code, 0)
        self.assertTrue((dest / "run.json").exists())
        self.assertTrue((dest / "summary.json").exists())
        self.assertTrue((dest / "audit" / "frozen_hashes.json").exists())
        self.assertTrue((dest / "stages" / "audit.complete.json").exists())
        self.assertTrue((dest / "artifacts.jsonl").exists())
        run = json.loads((dest / "run.json").read_text())
        self.assertEqual(run["config_hash"], json.loads((dest / "summary.json").read_text())["config_hash"])
        self.assertFalse(run["allow_api"])
        dumped = json.dumps(run)
        self.assertNotIn("/Users/", dumped)
        self.assertNotIn("/home/", dumped)
        art = (dest / "artifacts.jsonl").read_text().strip().splitlines()
        self.assertGreaterEqual(len(art), 2)
        rec = json.loads(art[0])
        self.assertIn("config_hash", rec)
        self.assertIn("git_commit", rec)
        self.assertEqual(rec["config_hash"], run["config_hash"])
        env = json.loads((dest / "audit" / "environment.json").read_text())
        self.assertEqual(env["woais_run"]["config_hash"], run["config_hash"])
        if run["git"].get("available"):
            self.assertIsNotNone(run["git"]["commit"])
            self.assertEqual(env["woais_run"]["git_commit"], run["git"]["commit"])
        else:
            self.assertIsNone(run["git"].get("commit"))

    def test_refuses_overwrite_without_resume_or_force(self):
        self.run_id = "unit_audit_clobber"
        dest = RUNS / self.run_id
        if dest.exists():
            shutil.rmtree(dest)
        self.assertEqual(main(["audit", "--run-id", self.run_id, "--no-symlink"]), 0)
        code = main(["audit", "--run-id", self.run_id, "--no-symlink"])
        self.assertEqual(code, 2)

    def test_resume_skips_completed_audit(self):
        self.run_id = "unit_audit_resume"
        dest = RUNS / self.run_id
        if dest.exists():
            shutil.rmtree(dest)
        self.assertEqual(main(["audit", "--run-id", self.run_id, "--no-symlink"]), 0)
        code = main(["audit", "--run-id", self.run_id, "--no-symlink", "--resume", self.run_id])
        self.assertEqual(code, 0)
        summary = json.loads((dest / "summary.json").read_text())
        self.assertEqual(summary["stages"][0]["status"], "skipped")

    def test_resume_retries_failed_stage(self):
        self.run_id = "unit_audit_fail_resume"
        dest = RUNS / self.run_id
        if dest.exists():
            shutil.rmtree(dest)
        self.assertEqual(main(["audit", "--run-id", self.run_id, "--no-symlink"]), 0)
        marker = dest / "stages" / "audit.complete.json"
        payload = json.loads(marker.read_text())
        payload["status"] = "fail"
        marker.write_text(json.dumps(payload) + "\n")
        code = main(["audit", "--run-id", self.run_id, "--no-symlink", "--resume", self.run_id])
        self.assertEqual(code, 0)
        summary = json.loads((dest / "summary.json").read_text())
        self.assertEqual(summary["stages"][0]["status"], "ok")
        marker_after = json.loads(marker.read_text())
        self.assertEqual(marker_after["status"], "ok")


class TestWriteResultPolicy(unittest.TestCase):
    def test_forbid_raises(self):
        dest = RUNS / "_probe_forbid"
        dest.mkdir(parents=True, exist_ok=True)
        tokens = set_run_context(
            output_root=dest,
            overwrite_policy="forbid",
            artifact_hook=None,
            run_meta={"run_id": "probe", "git_commit": "abc", "config_hash": "def"},
        )
        try:
            write_result("a.json", {"n": 1})
            with self.assertRaises(ResultExistsError):
                write_result("a.json", {"n": 2})
        finally:
            reset_run_context(tokens)
            shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
