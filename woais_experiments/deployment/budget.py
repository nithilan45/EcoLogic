"""Worst-case spend estimate and runtime cost cap for deployment-real."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from woais_experiments.deployment.pricing import DeploymentPrices
from woais_experiments.stress.load_generator import CostCapExceeded, CostGuard

APPROACH_FRACTION = 0.95


def estimate_input_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    return max(1, int(len(text) / float(chars_per_token)))


def max_call_cost_usd(
    *,
    prices: DeploymentPrices,
    model: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> float:
    cost, _ = prices.realized_provider_cost(model, int(max_input_tokens), int(max_output_tokens))
    if cost is None:
        raise ValueError(f"cannot price worst-case call for {model!r}")
    return float(cost)


def worst_case_budget(
    prompts: Sequence[Mapping[str, Any]],
    *,
    prices: DeploymentPrices,
    strong_model: str,
    n_policies: int,
    n_workloads: int,
    max_output_tokens: int,
    max_retries: int,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    """Upper bound: every request uses the strong model, max tokens, and all retries."""
    max_in = max(estimate_input_tokens(str(p["text"]), chars_per_token=chars_per_token) for p in prompts)
    unit = max_call_cost_usd(
        prices=prices,
        model=strong_model,
        max_input_tokens=max_in,
        max_output_tokens=int(max_output_tokens),
    )
    attempts = 1 + max(0, int(max_retries))
    n_calls = int(len(prompts)) * int(n_policies) * int(n_workloads)
    n_attempts = n_calls * attempts
    worst = float(n_attempts) * unit
    return {
        "estimator": "strong_model_max_tokens_times_retries",
        "not_measured_cost": True,
        "strong_model": strong_model,
        "max_input_tokens_est": max_in,
        "max_output_tokens": int(max_output_tokens),
        "chars_per_token_estimate": float(chars_per_token),
        "max_retries": int(max_retries),
        "attempts_per_request": attempts,
        "n_queries": len(prompts),
        "n_policies": int(n_policies),
        "n_workloads": int(n_workloads),
        "n_planned_requests": n_calls,
        "n_planned_attempts": n_attempts,
        "unit_worst_usd": unit,
        "worst_case_usd": worst,
        "note": (
            "Worst-case uses the strong-model price, estimated input tokens "
            "(chars/4), max_tokens output, and every retry. This is an estimate "
            "used only for the pre-run cap, not a measured energy or cost result."
        ),
    }


def abort_if_over_budget(estimate: Mapping[str, Any], max_cost_usd: float) -> None:
    worst = float(estimate["worst_case_usd"])
    limit = float(max_cost_usd)
    if worst > limit:
        raise CostCapExceeded(
            f"worst-case ${worst:.6f} exceeds --max-cost-usd ${limit:.6f}; aborting before any paid call",
            realized_usd=0.0,
            projected_usd=worst,
            limit_usd=limit,
        )


def make_guard(
    *,
    paid: bool,
    max_cost_usd: float | None,
    estimate_usd_per_query: float | None,
    n_planned: int,
) -> CostGuard:
    guard = CostGuard(max_cost_usd, estimate_usd_per_query=estimate_usd_per_query, paid=paid)
    guard.add_planned(int(n_planned))
    return guard


def check_approach(guard: CostGuard, *, extra_requests: int = 1) -> None:
    """Stop when realized or projected spend reaches 95% of the cap."""
    if guard.limit is None:
        return
    snap = guard.snapshot()
    unit = float(snap.get("unit_cost_usd") or snap.get("estimate_usd_per_query") or 0.0)
    projected = float(snap["realized_usd"]) + max(0, int(extra_requests)) * unit
    approach = APPROACH_FRACTION * float(guard.limit)
    if float(snap["realized_usd"]) >= approach or projected >= approach:
        raise CostCapExceeded(
            f"budget approached (realized=${snap['realized_usd']:.6f}, "
            f"projected=${projected:.6f}, 95% of cap=${approach:.6f})",
            realized_usd=float(snap["realized_usd"]),
            projected_usd=projected,
            limit_usd=float(guard.limit),
        )
    guard.check_ahead(extra_requests)
