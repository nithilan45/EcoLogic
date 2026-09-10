"""RouteLLM Stage 9 tables from committed JSON. No clone, no BERT, no API."""

from __future__ import annotations

from woais_experiments.frozen import load_json
from woais_experiments.paths import ROOT


def load_s9() -> dict:
    return load_json(ROOT / "stage7_10" / "s9_static_baselines.json")


def load_s9_generalization() -> dict:
    return load_json(ROOT / "stage7_10" / "external_generalization.json")


def summarize_s9(s9: dict | None = None) -> dict:
    s9 = s9 if s9 is not None else load_s9()
    interior = [r for r in s9["sweep"] if 0.5 < r["strong_pct"] < 99.5]
    edges_frac = [r["router_minus_random_same_fraction_pp"] for r in interior]
    edges_cost = [r["router_minus_random_matched_cost_pp"] for r in interior]
    lost = [r["advantage_lost_to_cost_matching_pp"] for r in interior]
    return {
        "source": s9["source"],
        "n_items": s9["n_items"],
        "always_weak_pct": s9["accuracy"]["always_weak_pct"],
        "always_strong_pct": s9["accuracy"]["always_strong_pct"],
        "n_interior": len(interior),
        "n_interior_beating_matched_cost": s9["n_interior_beating_matched_cost_mixture"],
        "n_interior_significant_p05": s9["n_interior_beating_matched_cost_mixture_p05"],
        "mean_edge_matched_fraction_pp": float(sum(edges_frac) / len(edges_frac)),
        "mean_edge_matched_cost_pp": float(sum(edges_cost) / len(edges_cost)),
        "mean_advantage_lost_to_cost_matching_pp": float(sum(lost) / len(lost)),
        "max_advantage_lost_to_cost_matching_pp": s9["max_advantage_lost_to_cost_matching_pp"],
        "sign_test_p": s9["sign_test_across_sweep"]["p_two_sided"],
        "naive_formula_exact_for_random_residual_usd": s9["naive_formula_exact_for_random_residual_usd"],
        "published_mean_edge_matched_cost_pp": s9["mean_router_edge_matched_cost_pp"],
        "sweep_interior": interior,
        "note": (
            "Re-tabulated from stage7_10/s9_static_baselines.json. "
            "RouteLLM is not re-run; /tmp/routellm_chk is not required."
        ),
    }


def summarize_generalization(blob: dict | None = None) -> dict:
    blob = blob if blob is not None else load_s9_generalization()
    gsm = blob.get("gsm8k", blob)
    # Keep portable fields; the JSON is nested.
    keep = {
        "n_items": gsm.get("n_items"),
        "weak_model": gsm.get("weak_model"),
        "strong_model": gsm.get("strong_model"),
        "within_model_cost_cv": gsm.get("within_model_cost_cv"),
        "release_audit": blob.get("release_audit"),
    }
    # Pull sweep-level bias numbers if present
    for key in ("max_naive_understate_pct", "sweep", "sign_flip"):
        if key in gsm:
            keep[key] = gsm[key]
    keep["note"] = (
        "Correction identity was already validated on RouteLLM GSM8K in Stage 9. "
        "This package does not re-tokenize or re-score that release."
    )
    return keep
