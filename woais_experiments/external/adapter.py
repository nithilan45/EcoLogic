"""Ingest heterogeneous routing tables into a canonical long panel.

Does not download data. Known in-repo sources (EcoLogic Stage 1–2 / Stage 7
matrices, committed RouteLLM Stage 9 JSON, and an optional local RouteLLM
GSM8K CSV if someone already placed one) are discovered by path existence.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.per_query_cost import PriceTable, load_price_table
from woais_experiments.external.schema import (
    LongPanel,
    SchemaError,
    as_float,
    as_str,
    check_model_count,
    rows_from_mappings,
)
from woais_experiments.frozen import ItemMatrix, load_s7_test_matrix, load_stage12_matrix
from woais_experiments.paths import ROOT, repo_rel
from woais_experiments.routing.policies import keyword_from_routing

ROUTELLM_WEAK = "mistralai/Mixtral-8x7B-Instruct-v0.1"
ROUTELLM_STRONG = "gpt-4-1106-preview"

# Local clones / drops only. Nothing here is fetched.
ROUTELLM_CSV_CANDIDATES = (
    Path(os.environ["ROUTELLM_RESPONSES"]) if os.environ.get("ROUTELLM_RESPONSES") else None,
    Path(os.environ["ROUTELLM_DATA"]) / "evals/gsm8k/gsm8k_responses.csv"
    if os.environ.get("ROUTELLM_DATA")
    else None,
    Path("/tmp/routellm_chk/routellm/evals/gsm8k/gsm8k_responses.csv"),
    ROOT / "woais_experiments" / "data" / "routellm" / "gsm8k_responses.csv",
    ROOT / "routellm" / "evals" / "gsm8k" / "gsm8k_responses.csv",
)


@dataclass
class WidePanel:
    """Complete-case query × model arrays aligned with ``names``."""

    names: tuple[str, ...]
    query_ids: tuple[str, ...]
    dataset: str
    quality: np.ndarray
    cost: np.ndarray
    latency_ms: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    assignment: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.quality.shape[0])

    @property
    def m(self) -> int:
        return int(self.quality.shape[1])


@dataclass
class DiscoveredSource:
    name: str
    kind: str
    available: bool
    path: str | None
    note: str
    extra: dict[str, Any] = field(default_factory=dict)


def _read_tabular(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        blob = json.loads(text)
        if isinstance(blob, list):
            return list(blob)
        if isinstance(blob, dict):
            for key in ("rows", "records", "data"):
                if isinstance(blob.get(key), list):
                    return list(blob[key])
        raise SchemaError(f"JSON at {path} is not a list of records")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    source: str = "records",
    latency_unit: str = "auto",
    prices: PriceTable | None = None,
    notes: Sequence[str] | None = None,
) -> LongPanel:
    """Long-format ingest. Tokens × prices may fill missing cost; see ``cost_source``."""
    rows = rows_from_mappings(records, dataset=dataset, latency_unit=latency_unit)
    extra_notes = list(notes or [])
    table = prices
    filled = 0
    for r in rows:
        if r.realized_cost is not None:
            continue
        if r.input_tokens is None or r.output_tokens is None:
            continue
        if table is None:
            table = load_price_table()
        if r.model not in table:
            continue
        r.realized_cost = table.inference_cost(r.model, r.input_tokens, r.output_tokens)
        r.cost_source = "price_table"
        filled += 1
    if filled:
        extra_notes.append(
            f"filled cost for {filled} rows from committed USD/1M rates "
            "(cost_source=price_table, not metered realized_cost)"
        )
    panel = LongPanel(rows=rows, dataset=dataset, source=source, notes=extra_notes)
    check_model_count(panel.models)
    return panel


def from_path(
    path: Path | str,
    *,
    dataset: str | None = None,
    latency_unit: str = "auto",
    prices: PriceTable | None = None,
) -> LongPanel:
    path = Path(path)
    recs = _read_tabular(path)
    ds = dataset or path.stem
    if _looks_like_routellm_wide(recs):
        return from_routellm_responses(recs, dataset=ds, source=repo_rel(path), prices=prices)
    return normalize_records(
        recs, dataset=ds, source=repo_rel(path), latency_unit=latency_unit, prices=prices
    )


def _looks_like_routellm_wide(records: Sequence[Mapping[str, Any]]) -> bool:
    if not records:
        return False
    keys = {_k.lower() for _k in records[0].keys()}
    return ROUTELLM_WEAK.lower() in keys and ROUTELLM_STRONG.lower() in keys


def from_routellm_responses(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str = "routellm_gsm8k",
    source: str = "routellm_csv",
    prices: PriceTable | None = None,
    weak: str = ROUTELLM_WEAK,
    strong: str = ROUTELLM_STRONG,
) -> LongPanel:
    """Melt RouteLLM's wide GSM8K CSV (correctness columns named by model).

    Response text is ignored. Cost is filled only when token columns exist or a
    committed price can be applied to token counts. Tokenizers are not invoked.
    """
    long_rows: list[dict[str, Any]] = []
    for i, raw in enumerate(records):
        row = {str(k): v for k, v in raw.items()}
        lower = {k.lower(): (k, v) for k, v in row.items()}
        qid = as_str(row.get("query_id") or row.get("item_id")) or f"gsm8k-{i:04d}"
        score = as_float(row.get("router_score") or row.get("win_rate"))
        assign = as_str(row.get("router_assignment"))
        for model in (weak, strong):
            orig_key, val = lower.get(model.lower(), (None, None))
            rec: dict[str, Any] = {
                "query_id": qid,
                "dataset": dataset,
                "model": model,
                "correctness": val,
                "router_score": score,
                "router_assignment": assign,
            }
            for kind, aliases in (
                ("input_tokens", (f"{model}_input_tokens", f"{orig_key}_input_tokens", "input_tokens")),
                ("output_tokens", (f"{model}_output_tokens", f"{orig_key}_output_tokens", "tok_out")),
                ("cost", (f"cost_{'weak' if model == weak else 'strong'}", f"{model}_cost")),
            ):
                for a in aliases:
                    if a and a in row and row[a] not in (None, ""):
                        rec[kind] = row[a]
                        break
            # tok_out_weak / tok_out_strong from Stage 9 reconstruction if present
            side = "weak" if model == weak else "strong"
            if rec.get("output_tokens") is None and row.get(f"tok_out_{side}") not in (None, ""):
                rec["output_tokens"] = row[f"tok_out_{side}"]
            long_rows.append(rec)
    notes = [
        "RouteLLM wide CSV melted to long format. Response text was not tokenized "
        "(no download, no tiktoken/transformers). Cost is present only if the file "
        "already carries token or cost columns."
    ]
    return normalize_records(
        long_rows, dataset=dataset, source=source, prices=prices, notes=notes
    )


def from_item_matrix(
    matrix: ItemMatrix,
    *,
    dataset: str,
    source: str,
    assignment_tiers: Mapping[Any, int] | None = None,
    latency_unit: str = "s",
) -> LongPanel:
    """EcoLogic frozen ItemMatrix → canonical long panel (2–3 eval models)."""
    tiers = sorted(matrix.model_of)
    names = [matrix.model_of[t] for t in tiers]
    check_model_count(names)
    assign_by_qid: dict[str, str] = {}
    if assignment_tiers is not None:
        for qid, t in assignment_tiers.items():
            assign_by_qid[str(qid)] = matrix.model_of[int(t)]
    records: list[dict[str, Any]] = []
    for qid in matrix.item_ids:
        routed = assign_by_qid.get(qid)
        for t in tiers:
            key = (t, qid)
            rec = {
                "query_id": qid,
                "dataset": dataset,
                "model": matrix.model_of[t],
                "correctness": bool(matrix.correct[key]),
                "input_tokens": int(matrix.prompt_tokens[key]),
                "output_tokens": int(matrix.completion_tokens[key]),
                "cost": float(matrix.usd[key]),
                "latency_s": matrix.latency_s.get(key),
                "router_assignment": routed,
            }
            records.append(rec)
    return normalize_records(
        records,
        dataset=dataset,
        source=source,
        latency_unit=latency_unit,
        notes=[f"from ItemMatrix n={matrix.n} models={names}"],
    )


def to_wide(
    panel: LongPanel,
    *,
    require_cost: bool = True,
    require_quality: bool = True,
) -> WidePanel:
    """Pivot long rows to (n_queries, n_models). Incomplete queries are dropped."""
    names = panel.models
    col = {m: j for j, m in enumerate(names)}
    qids = list(panel.query_ids)
    n, m = len(qids), len(names)
    quality = np.full((n, m), np.nan)
    cost = np.full((n, m), np.nan)
    latency = np.full((n, m), np.nan)
    tin = np.full((n, m), np.nan)
    tout = np.full((n, m), np.nan)
    row_of = {q: i for i, q in enumerate(qids)}
    for r in panel.rows:
        i, j = row_of[r.query_id], col[r.model]
        if r.quality is not None:
            quality[i, j] = r.quality
        if r.realized_cost is not None:
            cost[i, j] = r.realized_cost
        if r.latency_ms is not None:
            latency[i, j] = r.latency_ms
        if r.input_tokens is not None:
            tin[i, j] = r.input_tokens
        if r.output_tokens is not None:
            tout[i, j] = r.output_tokens
    keep = np.ones(n, dtype=bool)
    if require_quality:
        keep &= np.all(np.isfinite(quality), axis=1)
    if require_cost:
        keep &= np.all(np.isfinite(cost), axis=1)
    notes = list(panel.notes)
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"dropped {dropped} incomplete queries (missing quality/cost cells)")
    quality, cost, latency = quality[keep], cost[keep], latency[keep]
    tin, tout = tin[keep], tout[keep]
    kept_ids = tuple(q for q, ok in zip(qids, keep) if ok)
    if quality.shape[0] == 0:
        raise SchemaError("no complete query × model cells after filtering")
    assign_map = panel.assignment_map()
    choice = None
    if assign_map:
        choice = np.full(len(kept_ids), -1, dtype=int)
        for i, qid in enumerate(kept_ids):
            name = assign_map.get(qid)
            if name is None:
                continue
            if name not in col:
                raise SchemaError(f"router_assignment {name!r} is not in {names}")
            choice[i] = col[name]
        if np.any(choice < 0):
            n_miss = int(np.sum(choice < 0))
            notes.append(f"{n_miss} complete queries have no router_assignment")
            ok = choice >= 0
            quality, cost, latency = quality[ok], cost[ok], latency[ok]
            tin, tout = tin[ok], tout[ok]
            kept_ids = tuple(q for q, flag in zip(kept_ids, ok) if flag)
            choice = choice[ok]
    check_model_count(names)
    return WidePanel(
        names=names,
        query_ids=kept_ids,
        dataset=panel.dataset,
        quality=quality,
        cost=cost,
        latency_ms=latency,
        input_tokens=tin,
        output_tokens=tout,
        assignment=choice,
        notes=notes,
    )


def assignment_dict(wide: WidePanel) -> dict[str, str] | None:
    if wide.assignment is None:
        return None
    return {wide.query_ids[i]: wide.names[int(wide.assignment[i])] for i in range(wide.n)}


def find_routellm_csv() -> Path | None:
    for path in ROUTELLM_CSV_CANDIDATES:
        if path is None:
            continue
        if path.is_file():
            return path
    return None


def discover_sources() -> list[DiscoveredSource]:
    """Inventory in-repo / already-local routing datasets. Never downloads."""
    out: list[DiscoveredSource] = []
    s9 = ROOT / "stage7_10" / "s9_static_baselines.json"
    out.append(
        DiscoveredSource(
            name="routellm_s9_committed",
            kind="summary",
            available=s9.is_file(),
            path=repo_rel(s9) if s9.is_file() else None,
            note=(
                "Committed Stage 9 JSON (aggregates only). Per-query GSM8K responses "
                "are not in this repository and are not downloaded."
            ),
        )
    )
    csv_path = find_routellm_csv()
    out.append(
        DiscoveredSource(
            name="routellm_gsm8k_responses",
            kind="panel",
            available=csv_path is not None,
            path=repo_rel(csv_path) if csv_path else None,
            note=(
                "Optional local RouteLLM GSM8K CSV. Searched ROUTELLM_RESPONSES, "
                "ROUTELLM_DATA, /tmp/routellm_chk, and woais_experiments/data/routellm/. "
                "Not downloaded."
            ),
        )
    )
    try:
        matrix = load_stage12_matrix()
        out.append(
            DiscoveredSource(
                name="ecologic_stage12",
                kind="panel",
                available=True,
                path=repo_rel(ROOT / "raw_results" / "graded.jsonl"),
                note=(
                    f"Frozen EcoLogic Stage 1–2 item matrix, n={matrix.n}, "
                    f"models={[matrix.model_of[t] for t in sorted(matrix.model_of)]}"
                ),
                extra={"n": matrix.n, "n_models": len(matrix.model_of)},
            )
        )
    except FileNotFoundError as exc:
        out.append(
            DiscoveredSource(
                name="ecologic_stage12",
                kind="panel",
                available=False,
                path=None,
                note=str(exc),
            )
        )
    try:
        s7 = load_s7_test_matrix(sample_idx=0)
        out.append(
            DiscoveredSource(
                name="ecologic_stage7",
                kind="panel",
                available=True,
                path=repo_rel(ROOT / "stage7_10" / "s7_test_samples.csv.gz"),
                note=f"Frozen Stage 7 test matrix (sample_idx=0), n={s7.n}",
                extra={"n": s7.n, "n_models": len(s7.model_of)},
            )
        )
    except FileNotFoundError as exc:
        out.append(
            DiscoveredSource(
                name="ecologic_stage7",
                kind="panel",
                available=False,
                path=None,
                note=str(exc),
            )
        )
    return out


def load_source(name: str) -> LongPanel | dict[str, Any]:
    """Load a discovered source. Summaries return dicts; panels return LongPanel."""
    if name == "routellm_s9_committed":
        from woais_experiments.external.routellm import load_s9, load_s9_generalization, summarize_s9

        s9 = load_s9()
        return {
            "kind": "summary",
            "name": name,
            "panel_available": False,
            "summary": summarize_s9(s9),
            "generalization": load_s9_generalization().get("release_audit"),
            "n_models": 2,
            "models": [s9.get("weak_model", ROUTELLM_WEAK), s9.get("strong_model", ROUTELLM_STRONG)],
            "note": (
                "No per-query panel on disk. Full accounting/oracle/bootstrap need a "
                "long-format table; this source is the committed aggregate check only."
            ),
        }
    if name == "routellm_gsm8k_responses":
        path = find_routellm_csv()
        if path is None:
            raise FileNotFoundError("RouteLLM GSM8K CSV is not on disk (not downloaded)")
        return from_path(path, dataset="routellm_gsm8k")
    if name == "ecologic_stage12":
        from woais_experiments.frozen import load_stage12_routing

        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        assign = keyword_from_routing(routing, matrix.item_ids, "raw")
        return from_item_matrix(
            matrix,
            dataset="ecologic_stage12",
            source="raw_results/graded.jsonl+routing.json",
            assignment_tiers=assign,
        )
    if name == "ecologic_stage7":
        matrix = load_s7_test_matrix(sample_idx=0)
        return from_item_matrix(
            matrix,
            dataset="ecologic_stage7",
            source="stage7_10/s7_test_samples.csv.gz",
        )
    raise KeyError(f"unknown source {name!r}")
