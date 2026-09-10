"""Query-independent mix of cheap/strong matched to EcoLogic expected cost.

Matching uses labeled token *estimates* (chars/4 and stub output tokens) so a
static assignment can be built before any paid call. Realized cost is measured
later from the request log, not from this estimate.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from woais_experiments.deployment.pricing import DeploymentPrices
from woais_experiments.deployment.router import classify_prompt, model_for_tier


def estimated_cost_usd(
    prompt: str,
    serving_slug: str,
    prices: DeploymentPrices,
    *,
    output_tokens: int,
    chars_per_token: float = 4.0,
) -> float:
    inn = max(1, int(len(prompt) / chars_per_token))
    cost, _ = prices.realized_provider_cost(serving_slug, inn, int(output_tokens))
    if cost is None:
        raise ValueError(f"cannot estimate cost for {serving_slug!r}")
    return float(cost)


def cost_matched_assignments(
    prompts: Sequence[Mapping[str, Any]],
    *,
    prices: DeploymentPrices,
    models: Mapping[str, Any],
    classify_fn: Callable[[str], Any],
    cheap_slug: str,
    strong_slug: str,
    output_tokens: int,
    seed: int,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    """Bernoulli mix of cheap/strong with E[cost] = EcoLogic estimated cost."""
    eco_costs: list[float] = []
    cheap_costs: list[float] = []
    strong_costs: list[float] = []
    rows: list[dict[str, Any]] = []
    for item in prompts:
        prompt = str(item["text"])
        pid = str(item.get("id") or item.get("prompt_id") or len(rows))
        decision = classify_prompt(prompt, models=dict(models), classify_fn=classify_fn)
        eco = estimated_cost_usd(
            prompt, decision.selected_model, prices,
            output_tokens=output_tokens, chars_per_token=chars_per_token,
        )
        cheap_c = estimated_cost_usd(
            prompt, cheap_slug, prices,
            output_tokens=output_tokens, chars_per_token=chars_per_token,
        )
        strong_c = estimated_cost_usd(
            prompt, strong_slug, prices,
            output_tokens=output_tokens, chars_per_token=chars_per_token,
        )
        eco_costs.append(eco)
        cheap_costs.append(cheap_c)
        strong_costs.append(strong_c)
        rows.append({
            "prompt_id": pid,
            "text": prompt,
            "ecologic_model": decision.selected_model,
            "ecologic_tier": decision.recommended_tier,
        })

    c_eco = float(np.mean(eco_costs)) if eco_costs else 0.0
    c_cheap = float(np.mean(cheap_costs)) if cheap_costs else 0.0
    c_strong = float(np.mean(strong_costs)) if strong_costs else 0.0
    denom = c_strong - c_cheap
    if abs(denom) < 1e-18:
        f = 0.0
        feasible = abs(c_eco - c_cheap) < 1e-12
    else:
        f = (c_eco - c_cheap) / denom
        feasible = 0.0 <= f <= 1.0
        f = float(min(1.0, max(0.0, f)))

    rng = np.random.default_rng(int(seed))
    draws = rng.random(len(rows))
    assignments: dict[str, str] = {}
    for i, row in enumerate(rows):
        slug = strong_slug if draws[i] < f else cheap_slug
        assignments[row["prompt_id"]] = slug
        row["static_model"] = slug

    return {
        "measurement_type": "MEASURED",
        "assignment_policy": "query_independent_bernoulli",
        "token_estimate_source": "chars_div_4_and_stub_output_for_matching_only",
        "mix_fraction_strong": f,
        "cost_match_feasible": bool(feasible),
        "estimated_mean_cost_ecologic": c_eco,
        "estimated_mean_cost_cheap": c_cheap,
        "estimated_mean_cost_strong": c_strong,
        "seed": int(seed),
        "assignments": assignments,
        "rows": rows,
        "note": (
            "Static mix is query-independent. The fraction is chosen so expected "
            "cost under token estimates matches EcoLogic. Realized cost is measured later."
        ),
    }


def cheap_strong_slugs(models: Mapping[str, Any], roles: Mapping[str, str]) -> tuple[str, str]:
    cheap_key = str(roles.get("cheap") or "tier1")
    strong_key = str(roles.get("strong") or "tier3")
    cheap = models[cheap_key] if cheap_key in models else model_for_tier(dict(models), int(cheap_key.replace("tier", "")))
    strong = models[strong_key] if strong_key in models else model_for_tier(dict(models), int(strong_key.replace("tier", "")))
    return str(cheap["name"]), str(strong["name"])
