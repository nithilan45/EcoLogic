"""Generic two-model external-router panel. No EcoLogic training, no label routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.per_query_cost import PriceTable, load_price_table
from woais_experiments.external_routing.reconstruct_assignments import CHEAP, STRONG


@dataclass
class ExternalRouterPanel:
    """Aligned per-query arrays for one independently produced router."""

    router_name: str
    dataset: str
    query_ids: tuple[str, ...]
    cheap_model: str
    strong_model: str
    scores: np.ndarray
    cheap_quality: np.ndarray
    strong_quality: np.ndarray
    cheap_input_tokens: np.ndarray
    cheap_output_tokens: np.ndarray
    strong_input_tokens: np.ndarray
    strong_output_tokens: np.ndarray
    cheap_cost: np.ndarray
    strong_cost: np.ndarray
    notes: list[str] = field(default_factory=list)
    source: str = ""
    score_provenance: str = ""

    @property
    def n(self) -> int:
        return len(self.query_ids)

    def score_range(self) -> dict[str, float]:
        s = self.scores[np.isfinite(self.scores)]
        if s.size == 0:
            return {"min": float("nan"), "max": float("nan"), "n_finite": 0}
        return {"min": float(s.min()), "max": float(s.max()), "n_finite": int(s.size)}


def costs_from_tokens(
    *,
    cheap_model: str,
    strong_model: str,
    cheap_in: np.ndarray,
    cheap_out: np.ndarray,
    strong_in: np.ndarray,
    strong_out: np.ndarray,
    prices: PriceTable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    table = prices or load_price_table()
    c0 = np.array(
        [table.inference_cost(cheap_model, int(i), int(o)) for i, o in zip(cheap_in, cheap_out)],
        dtype=float,
    )
    c1 = np.array(
        [table.inference_cost(strong_model, int(i), int(o)) for i, o in zip(strong_in, strong_out)],
        dtype=float,
    )
    return c0, c1


def identify_cheap_strong_by_mean_cost(
    names: Sequence[str],
    mean_cost: Mapping[str, float],
) -> tuple[str, str]:
    """Cheapest / most expensive by unconditional mean USD. Not by quality."""
    cheap = min(names, key=lambda n: (float(mean_cost[n]), n))
    strong = max(names, key=lambda n: (float(mean_cost[n]), n))
    if cheap == strong:
        raise ValueError("cheap and strong collapsed to the same model")
    return cheap, strong


def from_arrays(
    *,
    router_name: str,
    dataset: str,
    query_ids: Sequence[str],
    scores: Sequence[float],
    cheap_model: str,
    strong_model: str,
    cheap_quality: Sequence[float],
    strong_quality: Sequence[float],
    cheap_input_tokens: Sequence[float],
    cheap_output_tokens: Sequence[float],
    strong_input_tokens: Sequence[float],
    strong_output_tokens: Sequence[float],
    cheap_cost: Sequence[float] | None = None,
    strong_cost: Sequence[float] | None = None,
    notes: Sequence[str] | None = None,
    source: str = "",
    score_provenance: str = "",
    prices: PriceTable | None = None,
) -> ExternalRouterPanel:
    ids = tuple(str(q) for q in query_ids)
    n = len(ids)
    wr = np.asarray(scores, dtype=float)
    q0 = np.asarray(cheap_quality, dtype=float)
    q1 = np.asarray(strong_quality, dtype=float)
    i0 = np.asarray(cheap_input_tokens, dtype=float)
    o0 = np.asarray(cheap_output_tokens, dtype=float)
    i1 = np.asarray(strong_input_tokens, dtype=float)
    o1 = np.asarray(strong_output_tokens, dtype=float)
    for arr, name in (
        (wr, "scores"),
        (q0, "cheap_quality"),
        (q1, "strong_quality"),
        (i0, "cheap_input_tokens"),
        (o0, "cheap_output_tokens"),
        (i1, "strong_input_tokens"),
        (o1, "strong_output_tokens"),
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} length {arr.shape} != n={n}")
    if cheap_cost is None or strong_cost is None:
        c0, c1 = costs_from_tokens(
            cheap_model=cheap_model,
            strong_model=strong_model,
            cheap_in=i0,
            cheap_out=o0,
            strong_in=i1,
            strong_out=o1,
            prices=prices,
        )
    else:
        c0 = np.asarray(cheap_cost, dtype=float)
        c1 = np.asarray(strong_cost, dtype=float)
    return ExternalRouterPanel(
        router_name=router_name,
        dataset=dataset,
        query_ids=ids,
        cheap_model=cheap_model,
        strong_model=strong_model,
        scores=wr,
        cheap_quality=q0,
        strong_quality=q1,
        cheap_input_tokens=i0,
        cheap_output_tokens=o0,
        strong_input_tokens=i1,
        strong_output_tokens=o1,
        cheap_cost=c0,
        strong_cost=c1,
        notes=list(notes or []),
        source=source,
        score_provenance=score_provenance,
    )


def load_generic_csv(path, *, router_name: str, dataset: str | None = None) -> ExternalRouterPanel:
    """Long or wide CSV with independent ``router_score`` plus both models' tokens/quality."""
    import csv
    from pathlib import Path

    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty panel {path}")
    keys = {k.lower() for k in rows[0]}
    if "cheap_quality" in keys and "strong_quality" in keys:
        return from_arrays(
            router_name=router_name,
            dataset=dataset or path.stem,
            query_ids=[r.get("query_id") or r.get("item_id") or str(i) for i, r in enumerate(rows)],
            scores=[r["router_score"] for r in rows],
            cheap_model=str(rows[0].get("cheap_model") or CHEAP),
            strong_model=str(rows[0].get("strong_model") or STRONG),
            cheap_quality=[r["cheap_quality"] for r in rows],
            strong_quality=[r["strong_quality"] for r in rows],
            cheap_input_tokens=[r["cheap_input_tokens"] for r in rows],
            cheap_output_tokens=[r["cheap_output_tokens"] for r in rows],
            strong_input_tokens=[r["strong_input_tokens"] for r in rows],
            strong_output_tokens=[r["strong_output_tokens"] for r in rows],
            cheap_cost=[r["cheap_cost"] for r in rows] if "cheap_cost" in rows[0] else None,
            strong_cost=[r["strong_cost"] for r in rows] if "strong_cost" in rows[0] else None,
            source=str(path),
            score_provenance="csv:router_score (supplied; not derived from labels)",
        )
    raise ValueError(
        f"{path} is not a generic external-router CSV "
        "(need cheap_quality, strong_quality, router_score, token columns)"
    )
