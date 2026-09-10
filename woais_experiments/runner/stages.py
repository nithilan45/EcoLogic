"""Stage implementations. All writes go through the active run directory."""

from __future__ import annotations

import json
from typing import Any, Callable

from woais_experiments.frozen import write_result
from woais_experiments.paths import CONFIGS
from woais_experiments.runner.session import STAGE_ORDER, RunSession

StageFn = Callable[[RunSession], dict[str, Any]]


def _load_stage12(session: RunSession) -> None:
    if session.matrix is not None:
        return
    from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
    from woais_experiments.routing.policies import build_stage12_policies

    session.matrix = load_stage12_matrix()
    session.routing = load_stage12_routing()
    session.policies = build_stage12_policies(
        session.matrix, session.routing, seed=session.seeds["random_policy"]
    )
    srv_name = (session.config.get("paths") or {}).get("serverless_json", "serverless.json")
    session.srv = json.loads((CONFIGS / srv_name).read_text())
    session.log.event("stage12.loaded", n=session.matrix.n)


def stage_audit(session: RunSession) -> dict[str, Any]:
    from woais_experiments.frozen import verify_frozen_hashes

    check = verify_frozen_hashes()
    write_result("audit/frozen_hashes.json", check)
    payload = {
        "ok": bool(check["ok"]),
        "n_files": check.get("n_files"),
        "n_mismatches": check.get("n_mismatches"),
        "n_missing": check.get("n_missing"),
        "git": session.git,
        "config_hash": session.config_hash,
        "seeds": session.seeds,
        "python": __import__("sys").version.split()[0],
        "allow_api": session.allow_api,
    }
    write_result("audit/environment.json", payload)
    return payload


def stage_accounting(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import (
        run_accounting,
        run_breakeven,
        run_per_query_accounting,
    )
    from woais_experiments.figures.plot import plot_naive_vs_true

    _load_stage12(session)
    accounting = run_accounting(
        session.matrix, session.routing, session.policies, session.exp
    )
    run_per_query_accounting(session.matrix, session.policies)
    breakeven = run_breakeven(session.matrix, session.policies, session.exp)
    plot_naive_vs_true(
        accounting["naive_vs_true"]["usd"], "figures/naive_vs_true_usd.png"
    )
    session.summaries["accounting"] = accounting
    session.summaries["breakeven"] = breakeven
    eco = next(
        r for r in breakeven["rows"]
        if r["router"] == "ecologic" and r["axis"] == "usd"
    )
    return {
        "n": session.matrix.n,
        "always_t2_dominates_ecologic": accounting["reproduction_vs_published"][
            "always_t2_dominates_ecologic"
        ],
        "cost_ratio_ecologic_over_t2": accounting["reproduction_vs_published"][
            "cost_ratio_ecologic_over_t2"
        ],
        "breakeven_ecologic_usd_status": eco["status"],
        "breakeven_ecologic_usd_net_savings": eco["net_savings"],
    }


def stage_static(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import run_accounting, run_routing
    from woais_experiments.figures.plot import plot_accuracy_vs_cost

    _load_stage12(session)
    accounting = session.summaries.get("accounting")
    if accounting is None:
        accounting = run_accounting(
            session.matrix, session.routing, session.policies, session.exp
        )
        session.summaries["accounting"] = accounting
    routing_out = run_routing(session.matrix, session.policies, accounting, session.exp)
    plot_accuracy_vs_cost(
        accounting["axis_usd"]["policies"],
        cost_key="cost",
        title="Stage 1–2 policies on measured USD (frozen generations)",
        relpath="figures/accuracy_vs_usd.png",
    )
    session.summaries["static"] = routing_out
    return {
        "regret_identity_ecologic_usd_reconciles": routing_out[
            "regret_ecologic_vs_oracle_usd"
        ]["reconciles"],
        "accuracy_optimal_static": routing_out[
            "accuracy_optimal_static_at_ecologic_budget"
        ],
    }


def stage_oracle(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import run_oracle_bounds

    _load_stage12(session)
    bounds = run_oracle_bounds(session.matrix, session.policies)
    session.summaries["oracle"] = bounds
    return bounds


def stage_latency(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import (
        run_latency,
        run_latency_frontier,
        run_s7_latency_spotcheck,
        run_serverless,
    )
    from woais_experiments.figures.plot import plot_latency_cdf

    _load_stage12(session)
    latency = run_latency(session.matrix, session.policies)
    frontier = run_latency_frontier(session.matrix, session.policies, session.exp)
    serverless = run_serverless(
        session.matrix, session.policies, session.exp, session.srv
    )
    s7 = run_s7_latency_spotcheck()
    plot_latency_cdf(session.matrix, "figures/latency_cdf_by_tier.png")
    session.summaries["latency"] = latency
    return {
        "ecologic_mean_s": latency["policies"]["ecologic"]["mean"],
        "always_t2_mean_s": latency["policies"]["always_t2"]["mean"],
        "direct_cheap_p50_ms": frontier["direct_cheap_p50_ms"],
        "router_plus_selected_p50_ms": frontier["router_plus_selected_p50_ms"],
        "s7_available": s7.get("available"),
        "serverless_honesty": serverless.get("honesty"),
    }


def stage_workload(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import run_simulated_workloads, run_workloads

    _load_stage12(session)
    frozen = run_workloads(session.matrix)
    simulated = run_simulated_workloads()
    return {
        "stage12_mix": frozen["stage12_mix"]["counts"],
        "simulated_n_runs": simulated["n_runs"],
        "SIMULATED": True,
    }


def stage_external(session: RunSession) -> dict[str, Any]:
    from woais_experiments.run_offline import run_external

    ext_cfg = (session.config.get("stages") or {}).get("external") or {}
    if ext_cfg.get("download") and not session.allow_api:
        raise RuntimeError("external.download requires --allow-api")
    payload = run_external(session.exp)
    return {
        "routellm_mean_edge_matched_cost_pp": payload["s9_static_baselines"][
            "mean_edge_matched_cost_pp"
        ],
        "n_items": payload["s9_static_baselines"]["n_items"],
        "downloaded": payload["adapter"].get("downloaded"),
    }


def stage_robustness(session: RunSession) -> dict[str, Any]:
    from woais_experiments.routing.robustness import run_stage12 as run_robustness_stage12

    _load_stage12(session)
    boot = session.config.get("bootstrap") or {}
    payload = run_robustness_stage12(
        session.matrix,
        session.policies["ecologic"],
        n_perm=int(boot.get("n_perm", 199)),
        n_bootstrap=int(boot.get("n_bootstrap_robustness", 50)),
        seed=session.seeds["monte_carlo"],
    )
    return {
        "baseline_status": payload["baseline_status"],
        "n_hull_flips": payload["summary"]["n_hull_flips"],
        "n_claim_reversed": payload["summary"]["n_hull_claim_reversed"],
        "claim_survives": payload["summary"][
            "primary_conclusion_survives_hull_perturbations"
        ],
    }


STAGES: dict[str, StageFn] = {
    "audit": stage_audit,
    "accounting": stage_accounting,
    "static": stage_static,
    "oracle": stage_oracle,
    "latency": stage_latency,
    "workload": stage_workload,
    "external": stage_external,
    "robustness": stage_robustness,
}


def stages_for_command(command: str) -> tuple[str, ...]:
    if command == "all":
        return STAGE_ORDER
    if command == "validate-artifact":
        raise KeyError("validate-artifact is not a reconstruction stage")
    if command not in STAGES:
        raise KeyError(command)
    return (command,)
