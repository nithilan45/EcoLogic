"""Turn independently produced router scores into assignments.

The evaluated route is a function of ``(score, threshold)`` only.
Quality, cost, and correctness are never arguments of ``route_by_score``.
Oracle routing is a separate function, labeled ``ORACLE_ASSIGNMENT``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

EXTERNAL_LEARNED_ROUTER = "EXTERNAL_LEARNED_ROUTER"
ORACLE_ASSIGNMENT = "ORACLE_ASSIGNMENT"

CHEAP = "cheap"
STRONG = "strong"


def route_by_score(
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    cheap: str = CHEAP,
    strong: str = STRONG,
) -> np.ndarray:
    """RouteLLM / controller rule: ``score >= threshold`` → strong, else cheap.

    This function has no quality/cost/label argument on purpose.
    """
    wr = np.asarray(scores, dtype=float)
    if wr.ndim != 1:
        raise ValueError("scores must be 1-D")
    out = np.full(wr.shape, cheap, dtype=object)
    out[np.isfinite(wr) & (wr >= float(threshold))] = strong
    return out


def oracle_route(
    cheap_quality: Sequence[float] | np.ndarray,
    strong_quality: Sequence[float] | np.ndarray,
    cheap_cost: Sequence[float] | np.ndarray,
    strong_cost: Sequence[float] | np.ndarray,
    *,
    cheap: str = CHEAP,
    strong: str = STRONG,
) -> np.ndarray:
    """Hindsight: cheapest correct model; if neither is correct, the cheaper one.

    Labeled ``ORACLE_ASSIGNMENT``. Not an evaluated deployable router.
    """
    q0 = np.asarray(cheap_quality, dtype=float) > 0.5
    q1 = np.asarray(strong_quality, dtype=float) > 0.5
    c0 = np.asarray(cheap_cost, dtype=float)
    c1 = np.asarray(strong_cost, dtype=float)
    cheaper_is_cheap = c0 <= c1
    pick_cheap = np.where(q0, True, np.where(q1, False, cheaper_is_cheap))
    out = np.where(pick_cheap, cheap, strong).astype(object)
    return out


def assignment_kind(name: str) -> str:
    if name == ORACLE_ASSIGNMENT or str(name).startswith("oracle"):
        return ORACLE_ASSIGNMENT
    return EXTERNAL_LEARNED_ROUTER


def as_side_index(assign: Sequence[str], *, cheap: str, strong: str) -> np.ndarray:
    """0 = cheap, 1 = strong."""
    out = np.zeros(len(assign), dtype=int)
    for i, a in enumerate(assign):
        if a == strong:
            out[i] = 1
        elif a != cheap:
            raise ValueError(f"assignment {a!r} is not {cheap!r} or {strong!r}")
    return out


def selected_from_side(
    side: np.ndarray,
    cheap_vals: np.ndarray,
    strong_vals: np.ndarray,
) -> np.ndarray:
    side = np.asarray(side, dtype=int)
    return np.where(side == 1, strong_vals, cheap_vals)


def panel_row(
    *,
    query_id: str,
    router_score: float | None,
    router_assignment: str,
    cheap_model: str,
    strong_model: str,
    cheap_quality: float,
    strong_quality: float,
    cheap_input_tokens: float | None,
    cheap_output_tokens: float | None,
    strong_input_tokens: float | None,
    strong_output_tokens: float | None,
    selected_model: str,
    selected_quality: float,
    selected_realized_cost: float,
    assignment_kind_value: str,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "router_score": router_score,
        "router_assignment": router_assignment,
        "cheap_model": cheap_model,
        "strong_model": strong_model,
        "cheap_quality": cheap_quality,
        "strong_quality": strong_quality,
        "cheap_input_tokens": cheap_input_tokens,
        "cheap_output_tokens": cheap_output_tokens,
        "strong_input_tokens": strong_input_tokens,
        "strong_output_tokens": strong_output_tokens,
        "selected_model": selected_model,
        "selected_quality": selected_quality,
        "selected_realized_cost": selected_realized_cost,
        "assignment_kind": assignment_kind_value,
    }
