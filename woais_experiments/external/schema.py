"""Canonical long-format schema for external routing benchmarks.

Incoming tables may carry any subset of source fields. After normalisation
every row is:

    query_id, dataset, model, quality, input_tokens, output_tokens,
    realized_cost, latency_ms

Optional router fields (``router_score``, ``router_assignment``) are stored
alongside the row but are not required. Quality is ``quality_score`` if
present, otherwise binary ``correctness``. Costs are taken from ``cost`` /
``realized_cost`` or from tokens × a committed price table.

The panel is *long*: one row per (query, model). Wide (query × model) arrays
are derived. Model count must be in [2, 50].
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Iterable, Mapping, Sequence

CANONICAL_COLUMNS = (
    "query_id",
    "dataset",
    "model",
    "quality",
    "input_tokens",
    "output_tokens",
    "realized_cost",
    "latency_ms",
)

OPTIONAL_COLUMNS = ("router_score", "router_assignment")

MIN_MODELS = 2
MAX_MODELS = 50

QUERY_ALIASES = (
    "query_id", "item_id", "qid", "example_id", "prompt_id", "question_id",
)
MODEL_ALIASES = ("model", "model_name", "selected_model", "llm", "engine", "tier_model")
INPUT_ALIASES = ("input_tokens", "prompt_tokens", "n_input_tokens", "in_tokens")
OUTPUT_ALIASES = ("output_tokens", "completion_tokens", "n_output_tokens", "out_tokens")
QUALITY_ALIASES = ("quality", "quality_score", "performance", "score", "winrate", "rating")
CORRECT_ALIASES = ("correctness", "correct", "is_correct", "ok", "label")
COST_ALIASES = ("realized_cost", "cost", "usd", "cost_usd", "price")
LATENCY_MS_ALIASES = ("latency_ms", "e2e_ms", "wall_ms")
LATENCY_S_ALIASES = ("latency_s", "latency_sec", "latency_seconds", "wall_s")
LATENCY_ALIASES = ("latency",)
ROUTER_SCORE_ALIASES = (
    "router_score", "win_rate", "p_strong", "router_win_rate", "strong_win_rate",
)
ROUTER_ASSIGN_ALIASES = (
    "router_assignment", "assignment", "routed_model", "selected", "t_hat",
)
DATASET_ALIASES = ("dataset", "benchmark", "split")
ROUTED_FLAG_ALIASES = ("routed", "is_selected", "chosen", "is_routed")

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}


class SchemaError(ValueError):
    """Incoming records cannot be mapped onto the canonical schema."""


def _norm_key(key: str) -> str:
    return str(key).strip().lower().replace(" ", "_")


def alias_map(aliases: Sequence[str]) -> dict[str, str]:
    canonical = aliases[0]
    return {_norm_key(a): canonical for a in aliases}


def first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, Any]:
    lower = {_norm_key(k): (k, v) for k, v in row.items()}
    for alias in aliases:
        hit = lower.get(_norm_key(alias))
        if hit is None:
            continue
        _, val = hit
        if val is None or val == "":
            continue
        return str(hit[0]), val
    return None, None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise SchemaError(
            "boolean is not a numeric measurement; use as_bool for quality flags"
        )
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"not a number: {value!r}") from exc
    if not math.isfinite(v):
        return None
    return v


def as_int(value: Any) -> int | None:
    v = as_float(value)
    if v is None:
        return None
    return int(round(v))


def as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1 or value == 1.0:
            return True
        if value == 0 or value == 0.0:
            return False
    s = str(value).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None


def as_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def latency_to_ms(value: Any, *, column: str | None, unit: str = "auto") -> float | None:
    v = as_float(value)
    if v is None:
        return None
    col = _norm_key(column or "")
    mode = (unit or "auto").lower()
    if mode in {"ms", "millisecond", "milliseconds"}:
        return v
    if mode in {"s", "sec", "second", "seconds"}:
        return v * 1000.0
    if mode != "auto":
        raise SchemaError(f"unknown latency unit {unit!r}")
    if col in {_norm_key(a) for a in LATENCY_MS_ALIASES}:
        return v
    if col in {_norm_key(a) for a in LATENCY_S_ALIASES}:
        return v * 1000.0
    if col == "latency" or col in {_norm_key(a) for a in LATENCY_ALIASES}:
        raise SchemaError(
            "column 'latency' is unit-ambiguous; pass latency_unit='ms' or latency_unit='s'"
        )
    raise SchemaError(
        f"cannot auto-detect latency unit for column {column!r}; pass latency_unit='ms' or 's'"
    )


def quality_from_row(row: Mapping[str, Any]) -> float | None:
    _, q = first_present(row, QUALITY_ALIASES)
    if q is not None:
        return as_float(q)
    _, c = first_present(row, CORRECT_ALIASES)
    if c is None:
        return None
    b = as_bool(c)
    if b is not None:
        return 1.0 if b else 0.0
    return as_float(c)


def check_model_count(models: Sequence[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(m) for m in models))
    if len(names) < MIN_MODELS:
        raise SchemaError(f"need at least {MIN_MODELS} models, got {len(names)}: {names}")
    if len(names) > MAX_MODELS:
        raise SchemaError(f"at most {MAX_MODELS} models supported, got {len(names)}")
    if any(not n for n in names):
        raise SchemaError("empty model name")
    return names


@dataclass
class CanonicalRow:
    query_id: str
    dataset: str
    model: str
    quality: float | None
    input_tokens: int | None
    output_tokens: int | None
    realized_cost: float | None
    latency_ms: float | None
    router_score: float | None = None
    router_assignment: str | None = None
    cost_source: str = "missing"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in (*CANONICAL_COLUMNS, *OPTIONAL_COLUMNS)}


@dataclass
class LongPanel:
    """Canonical long-format routing panel (one row per query × model)."""

    rows: list[CanonicalRow]
    dataset: str
    source: str
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rows:
            raise SchemaError("empty panel")
        models = check_model_count([r.model for r in self.rows])
        qids = tuple(dict.fromkeys(r.query_id for r in self.rows))
        seen: set[tuple[str, str]] = set()
        for r in self.rows:
            key = (r.query_id, r.model)
            if key in seen:
                raise SchemaError(f"duplicate cell {(r.query_id, r.model)}")
            seen.add(key)
        self.models = models
        self.query_ids = qids

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def to_records(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.rows]

    def to_csv(self) -> str:
        buf = StringIO()
        fields = list(CANONICAL_COLUMNS) + list(OPTIONAL_COLUMNS)
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in self.to_records():
            w.writerow({k: "" if rec[k] is None else rec[k] for k in fields})
        return buf.getvalue()

    def assignment_map(self) -> dict[str, str]:
        """query_id → routed model, from ``router_assignment`` when present."""
        out: dict[str, str] = {}
        for r in self.rows:
            name = as_str(r.router_assignment)
            if not name:
                continue
            prev = out.get(r.query_id)
            if prev is not None and prev != name:
                raise SchemaError(
                    f"conflicting router_assignment for {r.query_id}: {prev} vs {name}"
                )
            out[r.query_id] = name
        return out


def rows_from_mappings(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    latency_unit: str = "auto",
    default_model: str | None = None,
) -> list[CanonicalRow]:
    """Map heterogeneous dicts onto ``CanonicalRow`` (missing fields stay None)."""
    rows: list[CanonicalRow] = []
    for i, raw in enumerate(records):
        row = {str(k): v for k, v in raw.items()}
        _, qid_v = first_present(row, QUERY_ALIASES)
        qid = as_str(qid_v) or str(i)
        _, ds_v = first_present(row, DATASET_ALIASES)
        ds = as_str(ds_v) or dataset
        _, model_v = first_present(row, MODEL_ALIASES)
        model = as_str(model_v) or default_model
        if not model:
            raise SchemaError(f"row {qid} has no model")
        _, in_v = first_present(row, INPUT_ALIASES)
        _, out_v = first_present(row, OUTPUT_ALIASES)
        _, cost_v = first_present(row, COST_ALIASES)
        lat_col, lat_v = first_present(
            row, (*LATENCY_MS_ALIASES, *LATENCY_S_ALIASES, *LATENCY_ALIASES)
        )
        _, score_v = first_present(row, ROUTER_SCORE_ALIASES)
        _, assign_v = first_present(row, ROUTER_ASSIGN_ALIASES)
        _, flag_v = first_present(row, ROUTED_FLAG_ALIASES)
        assign = as_str(assign_v)
        routed = as_bool(flag_v)
        if assign is None and routed is True:
            assign = model
        metered = as_float(cost_v) if cost_v is not None else None
        rows.append(
            CanonicalRow(
                query_id=qid,
                dataset=ds,
                model=model,
                quality=quality_from_row(row),
                input_tokens=as_int(in_v) if in_v is not None else None,
                output_tokens=as_int(out_v) if out_v is not None else None,
                realized_cost=metered,
                latency_ms=latency_to_ms(lat_v, column=lat_col, unit=latency_unit)
                if lat_v is not None
                else None,
                router_score=as_float(score_v) if score_v is not None else None,
                router_assignment=assign,
                cost_source="metered" if metered is not None else "missing",
            )
        )
    return rows
