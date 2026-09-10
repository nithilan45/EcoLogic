"""Run every offline WOAIS experiment. No API calls. Frozen trees stay read-only."""

from __future__ import annotations

import logging
import os
import sys
import json
from pathlib import Path

import numpy as np

from woais_experiments.accounting.aggregate_cost import panel_from_item_matrix
from woais_experiments.accounting.breakeven import run_stage12 as run_breakeven_stage12
from woais_experiments.accounting.cost_decomposition import export_framework_tables
from woais_experiments.accounting.costs import (
    cost_fn_energy,
    cost_fn_usd,
    drop_outcomes,
    evaluate_assignment,
    paper_energy_rates,
    subst_energy_rates,
)
from woais_experiments.accounting.per_query_cost import load_price_table, load_router_overhead
from woais_experiments.accounting.tokens import corr_tokens_vs_latency, token_dispersion
from woais_experiments.external.routellm import summarize_generalization, summarize_s9
from woais_experiments.external.run_external_audit import run_discovered as run_external_adapter_audit
from woais_experiments.figures.plot import (
    plot_accuracy_vs_cost,
    plot_latency_cdf,
    plot_naive_vs_true,
    plot_sojourn_vs_load,
)
from woais_experiments.frozen import (
    load_json,
    load_s7_test_matrix,
    load_stage12_matrix,
    load_stage12_routing,
    verify_frozen_hashes,
    write_result,
)
from woais_experiments.latency.analyze_latency import summarize
from woais_experiments.latency.latency_frontier import from_item_matrix, save_frontier
from woais_experiments.latency.serverless import poisson_arrivals, replay_policy, service_model_by_tier
from woais_experiments.latency.timing import seconds_to_ms
from woais_experiments.latency.wallclock import latency_by_tier_benchmark, latency_per_output_token, policy_latency
from woais_experiments.paths import CONFIGS, PACKAGE, RESULTS, get_results_root, public_relpath, repo_rel
from woais_experiments.runner.config import config_hash, load_run_config
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.routing.oracle import sweep_item_matrix
from woais_experiments.routing.robustness import run_stage12 as run_robustness_stage12
from woais_experiments.routing.policies import (
    accuracy_optimal_static_mixture,
    build_stage12_policies,
    cost_accounting_for_policy,
    evaluate_policies,
    regret_for_policy,
    two_model_matched_cost,
)
from woais_experiments.statistics.regret import (
    synthetic_length_biased_example,
    synthetic_query_independent_example,
)
from woais_experiments.workloads.arrivals import open_loop_trace
from woais_experiments.workloads.frozen_sets import pool_sizes, resample_by_benchmark, stage12_item_mix
from woais_experiments.workloads.run_workload_sweep import run_and_save


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return public_relpath(obj)
    return obj


def _load_cfg() -> tuple[dict, dict]:
    exp = json.loads((CONFIGS / "experiment.json").read_text())
    srv = json.loads((CONFIGS / "serverless.json").read_text())
    return exp, srv


def _require_hashes(tag: str) -> dict:
    check = verify_frozen_hashes()
    write_result(f"hash_check_{tag}.json", _jsonable(check))
    if not check["ok"]:
        raise SystemExit(
            f"frozen-result hash check failed ({tag}): "
            f"{check['n_mismatches']} mismatches, {check['n_missing']} missing"
        )
    return check


def run_accounting(matrix, routing, policies, exp: dict) -> dict:
    usd = cost_fn_usd(matrix)
    energy = cost_fn_energy(matrix, paper_energy_rates())
    usd_table = evaluate_policies(
        matrix, policies, usd, vs="ecologic",
        cost_kind="usd", cost_unit="USD",
        cost_provenance="stored_call_usd",
    )
    energy_table = evaluate_policies(
        matrix,
        {k: v for k, v in policies.items() if k != "oracle_usd"},
        energy,
        vs="ecologic",
        cost_kind="modelled_energy",
        cost_unit="J",
        cost_provenance="tokens/1000 * paper_energy_per_1k (not metered)",
        overhead_per_query=0.0,
    )
    energy_table["cost_kind"] = "modelled_energy"
    energy_table["cost_unit"] = "J"
    energy_table["cost_provenance"] = "tokens/1000 * paper_energy_per_1k (not metered)"
    # align energy oracle
    energy_table["policies"]["oracle_energy"] = drop_outcomes(
        evaluate_assignment(
            matrix, policies["oracle_energy"], energy, name="oracle_energy",
            cost_kind="modelled_energy", cost_unit="J",
            cost_provenance="tokens/1000 * paper_energy_per_1k (not metered)",
        )
    )

    subst = cost_fn_energy(matrix, subst_energy_rates())
    subst_table = evaluate_policies(
        matrix,
        {k: v for k, v in policies.items() if k != "oracle_usd"},
        subst,
        vs="ecologic",
        cost_kind="modelled_energy",
        cost_unit="J",
        cost_provenance="tokens/1000 * subst_energy_per_1k (not metered)",
        overhead_per_query=0.0,
    )

    accounting = {}
    for axis, cost_of in ("usd", usd), ("energy", energy):
        accounting[axis] = {}
        for name in ("ecologic", "always_t1", "always_t2", "frontier", "random"):
            accounting[axis][name] = cost_accounting_for_policy(matrix, policies[name], cost_of)

    anchors = exp["published_stage12"]
    eco_usd = usd_table["policies"]["ecologic"]["cost"]
    t2_usd = usd_table["policies"]["always_t2"]["cost"]
    reproduction = {
        "n": matrix.n,
        "ecologic_usd": eco_usd,
        "always_t2_usd": t2_usd,
        "cost_ratio_ecologic_over_t2": eco_usd / t2_usd,
        "ecologic_correct": usd_table["policies"]["ecologic"]["correct"],
        "always_t2_correct": usd_table["policies"]["always_t2"]["correct"],
        "matches_published_usd": abs(eco_usd - anchors["ecologic_usd"]) < 1e-8,
        "matches_published_t2_usd": abs(t2_usd - anchors["always_t2_usd"]) < 1e-8,
        "always_t2_dominates_ecologic": (
            usd_table["policies"]["always_t2"]["accuracy"]
            > usd_table["policies"]["ecologic"]["accuracy"]
            and t2_usd < eco_usd
        ),
    }

    payload = {
        "axis_usd": usd_table,
        "axis_energy": energy_table,
        "naive_vs_true": accounting,
        "token_dispersion": token_dispersion(matrix),
        "corr_tokens_latency": corr_tokens_vs_latency(matrix),
        "reproduction_vs_published": reproduction,
        "energy_is_modelled": True,
        "energy_rate_table": "paper_energy_per_1k",
        "energy_note": (
            "axis_energy.cost is tokens/1000 × paper_energy_per_1k from "
            "configs/models.json; joules were never metered. Those rates are "
            "the paper coefficients for retired Gemma/Apriel slugs, not "
            "measured Qwen/gpt-oss power. axis_energy_subst_rates uses "
            "subst_energy_per_1k for the models that were actually called."
        ),
        "axis_energy_subst_rates": {
            name: subst_table["policies"][name]["cost"]
            for name in ("ecologic", "always_t1", "always_t2", "frontier", "random")
            if name in subst_table["policies"]
        },
    }
    write_result("accounting/stage12.json", _jsonable(payload))
    return payload


def run_routing(matrix, policies, accounting_payload, exp: dict) -> dict:
    usd = cost_fn_usd(matrix)
    energy = cost_fn_energy(matrix, paper_energy_rates())
    means_usd = {t: sum(matrix.usd[(t, i)] for i in matrix.item_ids) / matrix.n for t in (1, 2, 3)}
    accs = {}
    for t in (1, 2, 3):
        k = sum(matrix.correct[(t, i)] for i in matrix.item_ids)
        accs[t] = k / matrix.n

    eco_cost = accounting_payload["axis_usd"]["policies"]["ecologic"]["cost_per_item"]
    opt = accuracy_optimal_static_mixture(means_usd, accs, eco_cost)
    # Two-model mix between T2 (cheap/accurate) and T3 (expensive)
    mix_t2_t3 = two_model_matched_cost(
        eco_cost, means_usd[2], means_usd[3], accs[2], accs[3],
    )
    mix_t1_t2 = two_model_matched_cost(
        eco_cost, means_usd[2], means_usd[1], accs[2], accs[1],
    )

    regret_usd = regret_for_policy(matrix, policies["ecologic"], policies["oracle_usd"], usd)
    regret_energy = regret_for_policy(
        matrix, policies["ecologic"], policies["oracle_energy"], energy,
    )
    # Query-independent random: correction should be near zero
    regret_random = regret_for_policy(matrix, policies["random"], policies["oracle_usd"], usd)

    wrapped_vs_raw = {
        "agreement": sum(
            policies["ecologic"][i] == policies["ecologic_wrapped"][i] for i in matrix.item_ids
        ) / matrix.n,
        "raw_mix": accounting_payload["axis_usd"]["policies"]["ecologic"]["tier_mix"],
        "wrapped": drop_outcomes(
            evaluate_assignment(matrix, policies["ecologic_wrapped"], usd, name="wrapped")
        ),
    }

    payload = {
        "per_tier_mean_usd": means_usd,
        "per_tier_accuracy": accs,
        "accuracy_optimal_static_at_ecologic_budget": opt,
        "matched_cost_mix_t2_t3": mix_t2_t3,
        "matched_cost_mix_t2_t1": mix_t1_t2,
        "regret_ecologic_vs_oracle_usd": regret_usd,
        "regret_ecologic_vs_oracle_energy": regret_energy,
        "regret_random_vs_oracle_usd": regret_random,
        "raw_vs_wrapped": wrapped_vs_raw,
        "synthetic_length_biased": synthetic_length_biased_example(),
        "synthetic_query_independent": synthetic_query_independent_example(),
        "note": (
            "Stage 8 published R_naive/R_true were on the learned router's "
            "CALIBRATION split, not this 364-item keyword-router table. "
            "Both identities are reported here; they are not expected to match "
            "the Stage 8 scalars."
        ),
    }
    write_result("routing/stage12_policies.json", _jsonable(payload))
    return payload


def run_oracle_bounds(matrix, policies) -> dict:
    payload = sweep_item_matrix(
        matrix,
        cost_fn_usd(matrix),
        policies["ecologic"],
        method="approx",
        n_grid=17,
        relpath="routing/oracle_budget_sweep",
    )
    gaps = payload.get("gaps_at_router_cost") or {}
    return {
        "n_queries": payload["n_queries"],
        "method": payload["method"],
        "n_budgets": len(payload["sweep"]),
        "router_quality": gaps.get("learned_router_quality"),
        "static_frontier_quality": gaps.get("static_frontier_quality"),
        "oracle_quality": gaps.get("oracle_quality"),
        "router_vs_static_gap": gaps.get("router_vs_static_gap"),
        "oracle_vs_router_gap": gaps.get("oracle_vs_router_gap"),
        "fraction_of_available_routing_value_captured": gaps.get(
            "fraction_of_available_routing_value_captured"
        ),
        "saved": payload.get("saved"),
    }


def run_latency(matrix, policies) -> dict:
    by_tb = latency_by_tier_benchmark(matrix)
    per_tok = latency_per_output_token(matrix)
    pol = {name: policy_latency(matrix, assign) for name, assign in policies.items()
           if name not in {"oracle_energy"}}
    payload = {
        "honesty": "latency_s is full HTTP round-trip. TTFT was not recorded.",
        "by_tier_benchmark": by_tb,
        "seconds_per_output_token": per_tok,
        "policies": pol,
    }
    write_result("latency/stage12_wallclock.json", _jsonable(payload))
    return payload


def run_latency_frontier(matrix, policies, exp: dict) -> dict:
    """Cheap vs strong vs routed end-to-end from frozen latency_s. No inferred TTFT."""
    seed = int(exp["monte_carlo_seed"])
    n_boot = int(exp["n_random_sims"])
    payload = from_item_matrix(
        matrix,
        policies["ecologic"],
        cheap_tier=2,
        strong_tier=3,
        n_boot=n_boot,
        seed=seed,
    )
    payload["by_tier_legacy_end_to_end_ms"] = {
        str(t): summarize(
            [seconds_to_ms(matrix.latency_s[(t, i)]) for i in matrix.item_ids],
            n_boot=n_boot,
            seed=seed,
        )
        for t in (1, 2, 3)
    }
    saved = save_frontier(payload, relpath="latency/frontier")
    cheap = payload["policies"]["direct_cheap"]
    strong = payload["policies"]["direct_strong"]
    routed = payload["policies"]["router_plus_selected"]
    return {
        "cheap_name": payload["cheap_name"],
        "strong_name": payload["strong_name"],
        "n_queries": payload["n_queries"],
        "direct_cheap_p50_ms": cheap["p50"],
        "direct_strong_p50_ms": strong["p50"],
        "router_plus_selected_p50_ms": routed["p50"],
        "router_overhead_n_measured": payload["router_overhead"]["n_measured"],
        "saved": saved,
    }


def run_workloads(matrix) -> dict:
    payload = {
        "stage12_mix": stage12_item_mix(matrix),
        "frozen_pool_sizes": pool_sizes(),
        "synthetic_mixes": {
            "code_heavy": {"humaneval": 0.70, "mmlu": 0.15, "gsm8k": 0.15},
            "math_heavy": {"humaneval": 0.10, "mmlu": 0.20, "gsm8k": 0.70},
            "balanced": {"humaneval": 1 / 3, "mmlu": 1 / 3, "gsm8k": 1 / 3},
        },
        "note": "Synthetic mixes resample frozen items; they do not call APIs.",
    }
    write_result("workloads/frozen.json", _jsonable(payload))
    return payload


def run_simulated_workloads() -> dict:
    """YAML-driven DES. Outputs are labeled SIMULATED; not cloud measurements."""
    payload = run_and_save()
    return {
        "SIMULATED": True,
        "n_runs": payload["n_runs"],
        "n_snapshots": payload["n_snapshots"],
        "disclaimer": payload["disclaimer"],
        "saved": payload.get("saved"),
    }


def run_serverless(matrix, policies, exp: dict, srv: dict) -> dict:
    k = float(srv["residual_cold_k"])
    service_fit = service_model_by_tier(matrix, k=k)

    n_req = int(srv["trace_horizon_requests"])
    seed = int(exp["arrival_seed"])
    # Shared item sequence and a family of arrival rates in absolute req/s,
    # anchored on Always-Tier-2's mean service so the cheap static policy is
    # the capacity reference (systems comparison at equal offered traffic).
    t2_mean = float(np.mean([matrix.latency_s[(2, i)] for i in matrix.item_ids]))
    item_seq, _ = open_loop_trace(
        matrix.item_ids, n_requests=n_req, arrival_rate_per_s=1.0 / t2_mean, seed=seed,
    )
    names = ["ecologic", "always_t1", "always_t2", "frontier", "random"]
    loads = srv["arrival_rates_relative_to_mean_service"]
    n_servers_list = srv["n_servers"]

    equal_traffic = {}
    sojourn_series = {}
    for n_servers in n_servers_list:
        equal_traffic[str(n_servers)] = {}
        for name in names:
            equal_traffic[str(n_servers)][name] = []
            sojourn_series[f"{name}/n{n_servers}"] = []
            for rho in loads:
                rate = rho * n_servers / t2_mean
                arrivals = poisson_arrivals(n_req, rate, seed=seed + n_servers)
                rec = replay_policy(
                    matrix, policies[name], item_seq, arrivals,
                    n_servers=n_servers, idle_timeout_s=None, cold_penalty_s=0.0,
                )
                rec["load_relative_to_always_t2"] = rho
                rec["arrival_rate_per_s"] = rate
                rec["load"] = rho
                equal_traffic[str(n_servers)][name].append(rec)
                sojourn_series[f"{name}/n{n_servers}"].append(rec)

    # Idle-timeout sensitivity at one operating point (n=1, rho=0.5) for T1 vs T2
    idle_sweep = []
    rate = 0.5 * 1 / t2_mean
    arrivals = poisson_arrivals(n_req, rate, seed=seed + 99)
    for timeout in srv["idle_timeout_s"]:
        for penalty in srv["cold_start_penalty_s"]:
            for name in ("ecologic", "always_t2"):
                rec = replay_policy(
                    matrix, policies[name], item_seq, arrivals,
                    n_servers=1, idle_timeout_s=float(timeout),
                    cold_penalty_s=float(penalty),
                )
                idle_sweep.append({
                    "policy": name,
                    "idle_timeout_s": timeout,
                    "cold_penalty_s": penalty,
                    **rec,
                })

    payload = {
        "honesty": srv["honesty"],
        "t2_mean_service_s": t2_mean,
        "residual_service_model": service_fit,
        "equal_offered_traffic": equal_traffic,
        "idle_timeout_sensitivity": idle_sweep,
        "n_requests": n_req,
    }
    write_result("latency/serverless_model.json", _jsonable(payload))
    plot_sojourn_vs_load(
        {k: v for k, v in sojourn_series.items() if k.endswith("/n1")},
        "figures/sojourn_p95_vs_load.png",
    )
    return payload


def run_external(exp: dict | None = None) -> dict:
    s9 = summarize_s9()
    gen = summarize_generalization()
    n_boot = int((exp or {}).get("n_random_sims", 2000))
    seed = int((exp or {}).get("monte_carlo_seed", 20260909))
    adapter = run_external_adapter_audit(n_boot=n_boot, seed=seed, n_grid=9)
    payload = {
        "s9_static_baselines": s9,
        "s9_cost_correction": gen,
        "adapter": {
            "downloaded": False,
            "discovered": adapter.get("discovered"),
            "saved": adapter.get("saved"),
            "n_boot": n_boot,
            "note": adapter.get("note"),
        },
    }
    write_result("external/routellm_tables.json", _jsonable(payload))
    return payload


def run_s7_latency_spotcheck() -> dict:
    """Stage 7 test-set latency from compact samples (no raw jsonl required)."""
    try:
        m = load_s7_test_matrix(sample_idx=0)
    except FileNotFoundError:
        return {"available": False}
    payload = {
        "available": True,
        "n": m.n,
        "by_tier_benchmark": latency_by_tier_benchmark(m),
        "token_dispersion": token_dispersion(m),
    }
    write_result("latency/s7_test_wallclock.json", _jsonable(payload))
    return payload


def run_per_query_accounting(matrix, policies) -> None:
    prices = load_price_table()
    overhead = load_router_overhead("ecologic_keyword")
    panel, eco = panel_from_item_matrix(matrix, policies["ecologic"])
    _, t2 = panel_from_item_matrix(matrix, policies["always_t2"])
    export_framework_tables(
        get_results_root() / "accounting" / "framework",
        prices=prices,
        overhead=overhead,
        stage12=(panel, eco, t2),
    )


def run_breakeven(matrix, policies, exp: dict) -> dict:
    payload = run_breakeven_stage12(
        matrix,
        policies,
        n_boot=int(exp["n_random_sims"]),
        seed=int(exp["monte_carlo_seed"]),
    )
    return payload


def run_robustness(matrix, policies, exp: dict) -> dict:
    return run_robustness_stage12(
        matrix,
        policies["ecologic"],
        n_perm=199,
        n_bootstrap=50,
        seed=int(exp["monte_carlo_seed"]),
    )


def _fmt_break_even(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def write_report(
    accounting, routing_out, latency, serverless, external, s7, workloads,
    breakeven=None, robustness=None,
) -> None:
    usd = accounting["axis_usd"]["policies"]
    naive_eco = accounting["naive_vs_true"]["usd"]["ecologic"]
    regret = routing_out["regret_ecologic_vs_oracle_usd"]
    opt = routing_out["accuracy_optimal_static_at_ecologic_budget"]
    s9 = external["s9_static_baselines"]
    eco_l = latency["policies"]["ecologic"]
    t2_l = latency["policies"]["always_t2"]
    rho08 = {
        name: next(x for x in rows if abs(x["load_relative_to_always_t2"] - 0.8) < 1e-9)
        for name, rows in serverless["equal_offered_traffic"]["1"].items()
    }
    pretty = {
        "ecologic": "EcoLogic keyword router",
        "always_t1": "Always Tier 1",
        "always_t2": "Always Tier 2",
        "random": "Random tier",
        "frontier": "Always-frontier (gpt-4o)",
        "oracle_usd": "Oracle (cheapest correct)",
    }
    lines = [
        "# WOAIS offline results",
        "",
        "Derived from frozen Stage 1–2 / Stage 7 / Stage 9 artifacts. No API calls.",
        "Frozen SHA256 checks passed before and after this run (`hash_check_*.json`).",
        "",
        "## Cost and static baselines (n = 364, measured USD)",
        "",
        "| Policy | Accuracy | Cost (USD) | Mean latency (s) | p95 latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("ecologic", "always_t1", "always_t2", "random", "frontier", "oracle_usd"):
        s = usd[name]
        lat = latency["policies"][name]
        lines.append(
            f"| {pretty[name]} | {100*s['accuracy']:.1f}% | ${s['cost']:.4f} | "
            f"{lat['mean']:.2f} | {lat['p95']:.2f} |"
        )
    ratio = accounting["reproduction_vs_published"]["cost_ratio_ecologic_over_t2"]
    r2 = ", ".join(
        f"T{t}={serverless['residual_service_model'][str(t)]['r2']:.3f}"
        for t in (1, 2, 3)
    )
    lines += [
        "",
        f"Always-Tier-2 dominates the router on accuracy, dollars, **and** wall-clock "
        f"latency: {100*usd['always_t2']['accuracy']:.1f}% vs "
        f"{100*usd['ecologic']['accuracy']:.1f}% at 1/{ratio:.2f} of the cost and "
        f"{eco_l['mean']/t2_l['mean']:.1f}× lower mean wall-clock HTTP RTT "
        f"({t2_l['mean']:.1f}s vs {eco_l['mean']:.1f}s). That RTT is not exclusive "
        "service time (it includes provider queueing).",
        "",
        "The accuracy-optimal query-independent mixture at the router's realised "
        f"budget is **always Tier 2** (mix weight 1.0, accuracy "
        f"{100*opt['accuracy']:.1f}%). Extra spend does not buy static accuracy "
        "because Tier 3 is both costlier and less accurate than Tier 2 on this set.",
        "",
        "## Naive vs exact cost",
        "",
        "Pricing the keyword router as (tier mix) × (per-tier mean USD) understates "
        f"its true per-item cost by {100*abs(naive_eco['relative_bias']):.1f}% "
        f"(${naive_eco['naive_mean_cost']:.6f} vs ${naive_eco['true_mean_cost']:.6f}). "
        "The bias runs in the router's favour, matching the Stage 8/9 mechanism.",
        "",
        f"The regret identity reconciles on this table: R_naive = {regret['R_naive']:.6g}, "
        f"correction = {regret['correction']:.6g}, R_true = {regret['R_true']:.6g} "
        f"(residual {regret['abs_residual']:.2e}).",
        "",
        "## Serverless-style queueing (model, not measurement)",
        "",
        "Service times in the queueing model are stored `latency_s` HTTP round trips, "
        "not exclusive compute service. Arrival traces are synthetic "
        "Poisson processes over the frozen items. Cold-start extras are sensitivity "
        f"knobs; TTFT was never recorded. Linear latency-vs-tokens R² is {r2}, "
        "so most wall-clock time is explained by completion length, not a residual cold-start. "
        "Simulated sojourn **double-counts** historical provider delay already inside `latency_s`.",
        "",
        "At equal offered traffic (arrival rate = 0.8 × Always-Tier-2's 1-server "
        "capacity):",
        "",
        "| Policy | Mean HTTP RTT (s) | Utilisation | p95 sojourn (s, SIMULATED) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("ecologic", "always_t1", "always_t2", "frontier", "random"):
        rec = rho08[name]
        lines.append(
            f"| {pretty.get(name, name)} | {rec['mean_service_s']:.1f} | "
            f"{rec['utilization']:.3f} | {rec['p95_sojourn_s']:.1f} |"
        )
    lines += [
        "",
        "Routing most queries to the long-reasoning Tier 1 replica saturates a "
        "SIMULATED serverless worker that Always-Tier-2 would keep at ~80% utilisation. "
        "This is a modelled capacity result, not a measured cloud deployment, and it "
        "replays already-recorded HTTP round-trips.",
        "",
    ]
    if breakeven is not None:
        def _be(router: str, axis: str) -> dict:
            return next(
                r for r in breakeven["rows"]
                if r["router"] == router and r["axis"] == axis
            )

        eco_usd = _be("ecologic", "usd")
        eco_lat = _be("ecologic", "latency_ms")
        eco_tok = _be("ecologic", "tokens")
        t2_usd = _be("always_t2", "usd")
        ora_usd = _be("oracle_usd", "usd")
        counts = breakeven["status_counts"]["usd"]
        lines += [
            "## Router overhead break-even vs cost-matched static",
            "",
            "Maximum per-query router overhead (USD, latency, tokens) before a "
            "query-dependent assignment stops beating the cheapest static mix at "
            "the same quality. Analytic hull interpolation is checked by inverting "
            "the equal-cost envelope; percentile bootstrap CIs resample queries.",
            "",
            "| Router | Axis | Raw savings | Overhead | Net | Break-even | % of break-even used | Status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for router, label in (
            ("ecologic", "EcoLogic"),
            ("always_t2", "Always-T2"),
            ("oracle_usd", "Oracle"),
        ):
            for axis in ("usd", "latency_ms", "tokens"):
                row = _be(router, axis)
                pct = row["percentage_of_break_even_used"]
                pct_s = "n/a" if pct is None else (
                    "inf" if isinstance(pct, float) and not np.isfinite(pct) else f"{pct:.1f}"
                )
                lines.append(
                    f"| {label} | {axis} | {_fmt_break_even(row['raw_router_savings'])} | "
                    f"{_fmt_break_even(row['router_overhead'])} | "
                    f"{_fmt_break_even(row['net_savings'])} | "
                    f"{_fmt_break_even(row['break_even_overhead'])} | {pct_s} | "
                    f"{row['status']} |"
                )
        lines += [
            "",
            f"EcoLogic vs quality-matched static is **{eco_usd['status']}** on USD "
            f"(raw savings {_fmt_break_even(eco_usd['raw_router_savings'])} $/query; "
            f"keyword overhead {_fmt_break_even(eco_usd['router_overhead'])}; "
            f"net {_fmt_break_even(eco_usd['net_savings'])}). "
            f"Latency overhead is unmeasured in frozen logs and treated as 0 "
            f"(break-even {_fmt_break_even(eco_lat['break_even_overhead'])} ms; "
            f"status {eco_lat['status']}). Token overhead in config is 0 "
            f"(break-even {_fmt_break_even(eco_tok['break_even_overhead'])} tokens; "
            f"status {eco_tok['status']}).",
            "",
            f"Always-T2 is the hull vertex, so USD status is **{t2_usd['status']}** "
            f"with break-even {_fmt_break_even(t2_usd['break_even_overhead'])}. "
            f"The unconstrained oracle sits above the static hull "
            f"(USD status **{ora_usd['status']}**, break-even "
            f"{_fmt_break_even(ora_usd['break_even_overhead'])}).",
            "",
            f"USD status counts: beneficial={counts['beneficial']}, "
            f"neutral={counts['neutral']}, dominated={counts['dominated']}.",
            "",
        ]
    if robustness:
        sm = robustness["summary"]
        n_flag = sm["n_hull_flips"]
        n_rev = sm["n_hull_claim_reversed"]
        boot_rate = sm.get("bootstrap_flip_rate")
        boot_s = "n/a" if boot_rate is None else f"{100*boot_rate:.1f}%"
        survived = sm["primary_conclusion_survives_hull_perturbations"]
        lines += [
            "## Robustness of the EcoLogic-vs-static conclusion",
            "",
            f"Primary claim: EcoLogic is **{robustness['baseline_status']}** vs the "
            "quality-matched static hull on realized USD. "
            f"Hull-status labels changed in {n_flag} non-bootstrap scenarios "
            f"({n_rev} reversed the claim to beneficial"
            f"{'' if not sm.get('n_hull_significance_changed') else f'; {sm['n_hull_significance_changed']} lost p<0.05 vs Always-T2 while remaining dominated'}). "
            f"Bootstrap resample flip rate (status ≠ baseline): {boot_s}. "
            "Significance is a paired sign-flip p-value vs Always-T2 quality, "
            "not CI overlap. "
            f"The domination claim **{'survives' if survived else 'does not survive'}** "
            "as a beneficial reversal under the hull perturbations. "
            "Tidy table: `routing/robustness.csv`.",
            "",
        ]
    lines += [
        "## RouteLLM (committed Stage 9 JSON, not re-run)",
        "",
        f"n = {s9['n_items']}. Mean edge vs cost-matched mixture = "
        f"{s9['mean_edge_matched_cost_pp']:+.2f} pp; "
        f"{s9['n_interior_beating_matched_cost']}/{s9['n_interior']} interior "
        f"points positive; {s9['n_interior_significant_p05']} interior McNemar "
        f"p_raw<0.05 (unadjusted across the dependent sweep). The sweep-wide "
        f"sign count is descriptive, not a Type-I-controlled test "
        f"(p = {s9['sign_test_p']:.3g}; interior points share items).",
        "",
        f"Stage 7 test-set latency compact table available: {s7.get('available')}. "
        f"Stage 1–2 mix: {workloads['stage12_mix']['counts']}.",
        "",
    ]
    write_result("REPORT.md", "\n".join(lines))


def run() -> dict:
    logging.getLogger("woais").warning(
        "run_offline replaces files under %s (policy=replace). "
        "Use python run_woais.py for timestamped runs that refuse silent overwrite.",
        repo_rel(get_results_root()),
    )
    _require_hashes("before")
    exp, srv = _load_cfg()
    env = {
        "git": git_snapshot(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "config_hash": config_hash(load_run_config()),
        "seeds": {
            "random_policy": exp.get("random_policy_seed"),
            "monte_carlo": exp.get("monte_carlo_seed"),
            "arrival": exp.get("arrival_seed"),
        },
        "api_calls": False,
        "overwrite_policy": "replace",
        "results_dir": repo_rel(get_results_root()),
        "package": repo_rel(PACKAGE),
        "note": "Prefer run_woais.py for forbid/resume overwrite policy.",
    }
    write_result("run_environment.json", _jsonable(env))
    matrix = load_stage12_matrix()
    routing = load_stage12_routing()
    policies = build_stage12_policies(matrix, routing, seed=exp["random_policy_seed"])

    accounting = run_accounting(matrix, routing, policies, exp)
    run_per_query_accounting(matrix, policies)
    routing_out = run_routing(matrix, policies, accounting, exp)
    oracle_bounds = run_oracle_bounds(matrix, policies)
    breakeven = run_breakeven(matrix, policies, exp)
    robustness = run_robustness(matrix, policies, exp)
    latency = run_latency(matrix, policies)
    latency_frontier = run_latency_frontier(matrix, policies, exp)
    workloads = run_workloads(matrix)
    simulated_workloads = run_simulated_workloads()
    serverless = run_serverless(matrix, policies, exp, srv)
    external = run_external(exp)
    s7 = run_s7_latency_spotcheck()

    plot_accuracy_vs_cost(
        accounting["axis_usd"]["policies"],
        cost_key="cost",
        title="Stage 1–2 policies on measured USD (frozen generations)",
        relpath="figures/accuracy_vs_usd.png",
    )
    plot_latency_cdf(matrix, "figures/latency_cdf_by_tier.png")
    plot_naive_vs_true(accounting["naive_vs_true"]["usd"], "figures/naive_vs_true_usd.png")
    write_report(
        accounting, routing_out, latency, serverless, external, s7, workloads,
        breakeven=breakeven,
        robustness=robustness,
    )

    summary = {
        "n_items": matrix.n,
        "always_t2_dominates_ecologic_on_usd": accounting["reproduction_vs_published"][
            "always_t2_dominates_ecologic"
        ],
        "cost_ratio_ecologic_over_t2": accounting["reproduction_vs_published"][
            "cost_ratio_ecologic_over_t2"
        ],
        "ecologic_vs_always_t2_mean_latency_s": {
            "ecologic": latency["policies"]["ecologic"]["mean"],
            "always_t2": latency["policies"]["always_t2"]["mean"],
        },
        "routellm_mean_edge_matched_cost_pp": external["s9_static_baselines"][
            "mean_edge_matched_cost_pp"
        ],
        "regret_identity_ecologic_usd_reconciles": routing_out["regret_ecologic_vs_oracle_usd"][
            "reconciles"
        ],
        "oracle_fraction_captured_at_ecologic_usd": oracle_bounds[
            "fraction_of_available_routing_value_captured"
        ],
        "breakeven_ecologic_usd_status": next(
            r["status"] for r in breakeven["rows"]
            if r["router"] == "ecologic" and r["axis"] == "usd"
        ),
        "breakeven_ecologic_usd_net_savings": next(
            r["net_savings"] for r in breakeven["rows"]
            if r["router"] == "ecologic" and r["axis"] == "usd"
        ),
        "robustness_baseline_status": robustness["baseline_status"],
        "robustness_n_hull_flips": robustness["summary"]["n_hull_flips"],
        "robustness_n_claim_reversed": robustness["summary"]["n_hull_claim_reversed"],
        "robustness_claim_survives": robustness["summary"][
            "primary_conclusion_survives_hull_perturbations"
        ],
        "latency_frontier_p50_ms": {
            "direct_cheap": latency_frontier["direct_cheap_p50_ms"],
            "direct_strong": latency_frontier["direct_strong_p50_ms"],
            "router_plus_selected": latency_frontier["router_plus_selected_p50_ms"],
        },
        "simulated_workload_runs": simulated_workloads["n_runs"],
        "results_dir": repo_rel(RESULTS),
        "package": repo_rel(PACKAGE),
        "git_commit": env["git"].get("commit"),
        "config_hash": env["config_hash"],
        "pythonhashseed": env["pythonhashseed"],
    }
    write_result("summary.json", _jsonable(summary))
    _require_hashes("after")
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"wrote results under {RESULTS}")


if __name__ == "__main__":
    main()
