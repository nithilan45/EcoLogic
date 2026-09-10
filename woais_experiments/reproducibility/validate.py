"""Full reproducibility + anonymity pipeline. Writes only under results/reproducibility/."""

from __future__ import annotations

import json
from typing import Any

from woais_experiments.frozen import verify_frozen_hashes, write_result
from woais_experiments.paths import public_relpath
from woais_experiments.reproducibility.anonymity_scan import render_markdown, scan_tree as scan_anonymity
from woais_experiments.reproducibility.artifact_manifest import package_check
from woais_experiments.reproducibility.clean_clone_test import run_clean_clone
from woais_experiments.reproducibility.environment_capture import capture_environment
from woais_experiments.reproducibility.reproduce_tables import reconstruct, smoke_external_public
from woais_experiments.reproducibility.secret_scan import scan_tree as scan_secrets

PREFIX = "reproducibility"


def _write(name: str, payload: Any) -> str:
    dest = write_result(f"{PREFIX}/{name}", payload, clobber=True)
    return public_relpath(dest)


def run_validate_artifact(
    *,
    skip_clone: bool = False,
    clone_timeout_s: float = 1200.0,
) -> dict[str, Any]:
    env = capture_environment()
    hashes = verify_frozen_hashes()
    recon = reconstruct()
    external = smoke_external_public()
    secrets = scan_secrets()
    anonymity = scan_anonymity()
    pkg = package_check()

    clone: dict[str, Any]
    if skip_clone:
        clone = {
            "ok": False,
            "skipped": True,
            "exercised": False,
            "detail": (
                "clean-clone was not run. skipped is not a pass; a publication "
                "artifact must exercise the venv install without --skip-clone"
            ),
        }
    else:
        clone = run_clean_clone(timeout_s=clone_timeout_s)
        clone.setdefault("exercised", True)
        clone.setdefault("skipped", False)

    critical: list[str] = []
    if not hashes.get("ok"):
        critical.append("frozen_hashes")
    if not recon.get("ok"):
        critical.append("result_reconstruction")
    if not external.get("ok"):
        critical.append("external_smoke")
    if not secrets.get("ok"):
        critical.append("secret_scan")
    if anonymity.get("severity") == "critical" and not anonymity.get("ok"):
        critical.append("anonymity_scan")
    if not pkg.get("ok"):
        critical.append("package_check")
    if skip_clone:
        critical.append("clean_clone_skipped")
    elif not clone.get("ok"):
        critical.append("clean_clone")

    provenance = {
        "ok": recon.get("ok"),
        "records": recon.get("records"),
        "note": recon.get("note"),
        "environment": {
            "python": env.get("python"),
            "git_commit": (env.get("git") or {}).get("commit"),
            "seeds": env.get("random_seeds"),
            "config_hash": env.get("config_hash"),
        },
    }

    report = {
        "ok": len(critical) == 0,
        "critical": critical,
        "frozen_hashes": {
            "ok": hashes.get("ok"),
            "n_files": hashes.get("n_records"),
            "n_mismatches": hashes.get("n_mismatches"),
            "n_missing": hashes.get("n_missing"),
        },
        "reconstruction": {"ok": recon.get("ok"), "n_failed": recon.get("n_failed")},
        "external_smoke": external,
        "secret_scan": {
            "ok": secrets.get("ok"),
            "n_hits": secrets.get("n_hits"),
            "findings": secrets.get("findings"),
        },
        "anonymity_scan": {
            "ok": anonymity.get("ok"),
            "n_hits": anonymity.get("n_hits"),
            "n_files_requiring_review": anonymity.get("n_files_requiring_review"),
            "files_requiring_review": anonymity.get("files_requiring_review"),
        },
        "package_check": {
            "ok": pkg.get("ok"),
            "n_critical": pkg.get("n_critical"),
            "n_warnings": pkg.get("n_warnings"),
            "critical": pkg.get("critical"),
            "warnings": pkg.get("warnings")[:50],
        },
        "clean_clone": {
            "ok": clone.get("ok"),
            "skipped": clone.get("skipped", False),
            "exercised": clone.get("exercised", not clone.get("skipped", False)),
            "error": clone.get("error"),
            "checks_ok": (clone.get("checks") or {}).get("ok"),
        },
        "environment": env,
        "clone_environment": clone.get("clone_environment"),
    }

    written = {
        "environment.json": _write("environment.json", env),
        "result_provenance.json": _write("result_provenance.json", provenance),
        "secret_scan.json": _write("secret_scan.json", secrets),
        "anonymity_scan.json": _write("anonymity_scan.json", {
            k: anonymity[k] for k in anonymity if k != "findings"
        } | {"findings": anonymity.get("findings")}),
        "ANONYMITY_REPORT.md": _write("ANONYMITY_REPORT.md", render_markdown(anonymity)),
        "package_check.json": _write("package_check.json", pkg),
        "clean_clone.json": _write("clean_clone.json", {
            k: clone[k] for k in clone if k != "checks"
        } | {"checks": clone.get("checks")}),
        "validate.json": _write("validate.json", report),
    }
    report["written"] = written
    _write("validate.json", report)
    return report
