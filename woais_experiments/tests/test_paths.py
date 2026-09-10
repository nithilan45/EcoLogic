"""Repository-relative path serialization. No host-prefix allowlists."""

from __future__ import annotations

import unittest
from pathlib import Path

from woais_experiments.tests import ROOT  # noqa: F401
from woais_experiments.frozen import to_jsonable
from woais_experiments.paths import (
    REDACTED_ABSOLUTE,
    RESULTS,
    looks_like_home_absolute,
    public_relpath,
    relative_to_root,
    repo_rel,
)


def _join(*parts: str) -> str:
    """Build a POSIX absolute path without writing host-root literals in source."""
    return chr(47) + chr(47).join(parts)


def _win_drive(*parts: str) -> str:
    return "C:" + chr(47) + chr(47).join(parts)


def _win_backslash(*parts: str) -> str:
    return "C:" + chr(92) + chr(92).join(parts)


class TestRelativeToRoot(unittest.TestCase):
    def test_users_name_project(self):
        root = _join("Users", "name", "project")
        inner = _join("Users", "name", "project", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            relative_to_root(inner, root),
            "woais_experiments/results/foo.csv",
        )
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_home_name_project(self):
        root = _join("home", "name", "project")
        inner = _join("home", "name", "project", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_mnt_data_project(self):
        root = _join("mnt", "data", "EcoLogic-main")
        inner = _join("mnt", "data", "EcoLogic-main", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )
        self.assertEqual(
            repo_rel(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_tmp_project(self):
        root = _join("tmp", "project")
        inner = _join("tmp", "project", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_windows_drive_forward_slash(self):
        root = _win_drive("Users", "name", "project")
        inner = _win_drive("Users", "name", "project", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_windows_drive_backslash(self):
        root = _win_backslash("Users", "name", "project")
        inner = _win_backslash("Users", "name", "project", "woais_experiments", "results", "foo.csv")
        self.assertEqual(
            public_relpath(inner, root=root),
            "woais_experiments/results/foo.csv",
        )

    def test_outside_repo_is_redacted(self):
        root = _join("mnt", "data", "EcoLogic-main")
        self.assertEqual(
            public_relpath(_join("mnt", "data", "other", "secret.csv"), root=root),
            REDACTED_ABSOLUTE,
        )
        self.assertEqual(
            public_relpath(_join("Users", "name", "secret.json"), root=_join("home", "name", "project")),
            REDACTED_ABSOLUTE,
        )
        self.assertTrue(
            looks_like_home_absolute(_join("Users", "example_user", "secret.json"), root=root)
        )
        self.assertFalse(
            looks_like_home_absolute(
                _join("mnt", "data", "EcoLogic-main", "woais_experiments", "results", "foo.csv"),
                root=root,
            )
        )

    def test_actual_checkout_results_are_repo_relative(self):
        self.assertEqual(
            public_relpath(RESULTS / "summary.json"),
            "woais_experiments/results/summary.json",
        )
        self.assertEqual(
            repo_rel(RESULTS / "summary.json"),
            "woais_experiments/results/summary.json",
        )
        self.assertFalse(str(public_relpath(RESULTS / "foo.csv")).startswith("/"))

    def test_to_jsonable_relativizes_in_repo_absolutes(self):
        converted = to_jsonable(str(RESULTS / "ablations" / "index.json"))
        self.assertEqual(converted, "woais_experiments/results/ablations/index.json")
        self.assertFalse(str(converted).startswith(_join("Users")))
        self.assertFalse(str(converted).startswith(_join("home")))
        self.assertFalse(str(converted).startswith(_join("mnt")))

    def test_to_jsonable_redacts_foreign_absolutes(self):
        fake_home = chr(47).join(["", "Users", "example_user", "secret.json"])
        self.assertEqual(to_jsonable(fake_home), REDACTED_ABSOLUTE)
        self.assertEqual(
            to_jsonable(_join("mnt", "data", "unrelated", "secret.csv")),
            REDACTED_ABSOLUTE,
        )

    def test_no_host_prefix_allowlist_in_paths_module(self):
        from woais_experiments import paths as paths_mod
        src = Path(paths_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_HOME_ABS_PREFIXES", src)
        self.assertNotIn('"/Users/"', src)
        self.assertNotIn('"/home/"', src)
        self.assertNotIn('"/mnt/data/"', src)


if __name__ == "__main__":
    unittest.main()
