"""CLI for `python run_woais.py <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from typing import Any, Sequence

from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath, set_overwrite_policy
from woais_experiments.runner.config import load_run_config
from woais_experiments.runner.session import (
    COMMANDS,
    PrerequisiteError,
    apply_seeds,
    open_session,
    resolve_resume_dir,
    validate_prerequisites,
)
from woais_experiments.runner.stages import STAGES, stages_for_command


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_woais.py",
        description=(
            "Reproducible WOAIS experiment runner. Writes timestamped directories "
            "under woais_experiments/results/runs/. No paid API calls unless "
            "--allow-api is passed."
        ),
    )
    p.add_argument(
        "command",
        choices=COMMANDS,
        help="Stage to run, or 'all'",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML config (default: woais_experiments/configs/run.yaml)",
    )
    p.add_argument(
        "--allow-api",
        action="store_true",
        help="Permit paid provider API calls. Default: refuse.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--resume",
        nargs="?",
        const="LATEST",
        default=None,
        help="Resume a run directory (default: results/runs/latest)",
    )
    g.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts in a new or existing run dir",
    )
    p.add_argument("--run-id", default=None, help="Directory name under results/runs/")
    p.add_argument("--no-symlink", action="store_true", help="Do not update runs/latest")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging to the console",
    )
    p.add_argument(
        "--skip-clone",
        action="store_true",
        help="validate-artifact: skip the clean-venv clone (scans + reconstruction only)",
    )
    p.add_argument(
        "--clone-timeout-s",
        type=float,
        default=1200.0,
        help="validate-artifact: max seconds for the clean-clone step",
    )
    return p


def _print_summary(session, stage_rows: list[dict[str, Any]]) -> None:
    git = session.git.get("commit") or "unknown"
    dirty = " dirty" if session.git.get("dirty") else ""
    lines = [
        "",
        "WOAIS experiment summary",
        f"  run_id:       {session.run_id}",
        f"  command:      {session.command}",
        f"  git:          {git}{dirty}",
        f"  config_hash:  {session.config_hash[:16]}…",
        f"  seeds:        policy={session.seeds['random_policy']} "
        f"mc={session.seeds['monte_carlo']} arrival={session.seeds['arrival']}",
        f"  allow_api:    {session.allow_api}",
        f"  run_dir:      {public_relpath(session.run_dir)}",
        f"  latest_link:  {session.symlink_ok}",
        "  stages:",
    ]
    for row in stage_rows:
        lines.append(
            f"    {row['stage']:<12} {row['status']:<8} {row['elapsed_s']:.2f}s"
        )
    highlights = []
    for row in stage_rows:
        s = row.get("summary") or {}
        if row["stage"] == "accounting" and "always_t2_dominates_ecologic" in s:
            highlights.append(
                f"  always_t2_dominates: {s['always_t2_dominates_ecologic']}"
            )
            highlights.append(
                f"  cost_ratio_eco/t2:   {s['cost_ratio_ecologic_over_t2']:.4g}"
            )
        if row["stage"] == "robustness" and "claim_survives" in s:
            highlights.append(f"  robustness_survives: {s['claim_survives']}")
        if row["stage"] == "oracle" and "fraction_of_available_routing_value_captured" in s:
            highlights.append(
                f"  oracle_frac_captured: "
                f"{s['fraction_of_available_routing_value_captured']}"
            )
        if row["stage"] == "external" and "routellm_mean_edge_matched_cost_pp" in s:
            highlights.append(
                f"  routellm_edge_pp:    {s['routellm_mean_edge_matched_cost_pp']:+.3g}"
            )
    if highlights:
        lines.append("  highlights:")
        lines.extend(highlights)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)


def run_command(args: argparse.Namespace) -> int:
    cfg = load_run_config(args.config)
    resume_dir = resolve_resume_dir(args.resume) if args.resume else None
    if args.force and resume_dir is None and args.run_id:
        from woais_experiments.paths import RUNS
        cand = RUNS / args.run_id
        if cand.exists():
            resume_dir = cand

    try:
        session = open_session(
            command=args.command,
            config=cfg,
            allow_api=bool(args.allow_api),
            resume_dir=resume_dir,
            force=bool(args.force),
            run_id=args.run_id,
            no_symlink=bool(args.no_symlink),
            verbose=bool(args.verbose),
        )
    except PrerequisiteError as exc:
        sys.stderr.write(f"prerequisites failed: {exc}\n")
        return 2

    try:
        errors = validate_prerequisites(session, args.command)
        if errors:
            for e in errors:
                session.log.event("prereq.failed", level=logging.ERROR, error=e)
            raise PrerequisiteError("; ".join(errors))

        if args.allow_api:
            session.log.event(
                "api.allow_flag",
                level=logging.WARNING,
                note="--allow-api set; this tree still has no paid generation stages",
            )

        names = stages_for_command(args.command)
        rows: list[dict[str, Any]] = []
        for name in names:
            if session.overwrite == "resume" and session.is_complete(name):
                session.log.event("stage.skip", stage=name, reason="complete")
                marker = json.loads(session.complete_path(name).read_text())
                rows.append(
                    {
                        "stage": name,
                        "status": "skipped",
                        "elapsed_s": 0.0,
                        "summary": marker.get("summary"),
                    }
                )
                continue
            session.log.event("stage.start", stage=name)
            t0 = time.perf_counter()
            try:
                apply_seeds(session.seeds)
                # Resume keeps completed stages; leftover files from a failed
                # or interrupted stage must be replaced, not silently reused.
                if session.overwrite == "resume":
                    set_overwrite_policy("force")
                try:
                    summary = STAGES[name](session)
                finally:
                    if session.overwrite == "resume":
                        set_overwrite_policy("resume")
                elapsed = time.perf_counter() - t0
                session.mark_complete(name, summary, elapsed)
                session.log.event("stage.ok", stage=name, elapsed_s=round(elapsed, 3))
                rows.append(
                    {
                        "stage": name,
                        "status": "ok",
                        "elapsed_s": elapsed,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                session.log.event(
                    "stage.fail",
                    level=logging.ERROR,
                    stage=name,
                    error=str(exc),
                    elapsed_s=round(elapsed, 3),
                )
                session.mark_complete(
                    name,
                    {"error": str(exc), "traceback": traceback.format_exc()},
                    elapsed,
                    status="fail",
                )
                rows.append(
                    {
                        "stage": name,
                        "status": "fail",
                        "elapsed_s": elapsed,
                        "summary": {"error": str(exc)},
                    }
                )
                write_result(
                    "summary.json",
                    {
                        "run_id": session.run_id,
                        "ok": False,
                        "failed_stage": name,
                        "stages": rows,
                        "git_commit": session.git.get("commit"),
                        "config_hash": session.config_hash,
                    },
                    clobber=True,
                )
                _print_summary(session, rows)
                return 1

        session.index_tree()
        payload = {
            "run_id": session.run_id,
            "ok": True,
            "command": args.command,
            "git_commit": session.git.get("commit"),
            "git_dirty": session.git.get("dirty"),
            "config_hash": session.config_hash,
            "seeds": session.seeds,
            "allow_api": session.allow_api,
            "run_dir": public_relpath(session.run_dir),
            "stages": rows,
        }
        write_result("summary.json", payload, clobber=True)
        session.log.event("run.ok", n_stages=len(rows))
        _print_summary(session, rows)
        return 0
    except PrerequisiteError as exc:
        sys.stderr.write(f"prerequisites failed: {exc}\n")
        return 2
    finally:
        session.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-artifact":
        from woais_experiments.reproducibility.validate import run_validate_artifact

        report = run_validate_artifact(
            skip_clone=bool(args.skip_clone),
            clone_timeout_s=float(args.clone_timeout_s),
        )
        compact = {
            "ok": report.get("ok"),
            "critical": report.get("critical"),
            "written": report.get("written"),
            "anonymity_files": (report.get("anonymity_scan") or {}).get("n_files_requiring_review"),
            "secret_hits": (report.get("secret_scan") or {}).get("n_hits"),
            "reconstruction_ok": (report.get("reconstruction") or {}).get("ok"),
            "clean_clone_ok": (report.get("clean_clone") or {}).get("ok"),
            "clean_clone_skipped": (report.get("clean_clone") or {}).get("skipped"),
        }
        sys.stdout.write(json.dumps(compact, indent=2, default=str) + "\n")
        return 0 if report.get("ok") else 1
    return run_command(args)
