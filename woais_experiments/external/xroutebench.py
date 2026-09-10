"""Load the public xRouteBench routing panels from Hugging Face.

Downloads are cached under ``woais_experiments/data/xroutebench/hf_cache/``.
Config and split names are detected at runtime (not a hardcoded allowlist).
Missing tokens, latency, prices, and quality stay missing — they are never
filled from EcoLogic ``models.json``, from ``token_num``, or from estimates.

No paid model APIs. The Hugging Face Hub fetch is the public dataset only.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from woais_experiments.external.schema import (
    CANONICAL_COLUMNS,
    OPTIONAL_COLUMNS,
    CanonicalRow,
    LongPanel,
    MAX_MODELS,
    MIN_MODELS,
    SchemaError,
    as_float,
    as_int,
    as_str,
    check_model_count,
)
from woais_experiments.paths import PACKAGE

DEFAULT_REPO = os.environ.get("XROUTEBENCH_REPO", "ulab-ai/xRouteBench")
CACHE_DIR = PACKAGE / "data" / "xroutebench" / "hf_cache"
MILLION = 1_000_000.0
MIN_QUERIES = 10

# Original identity only. Never ``id`` (too generic) and never ``embedding_id``.
QUERY_ID_FIELDS = ("query_id", "task_id")
MODEL_FIELDS = ("model_name", "model")
QUALITY_FIELDS = ("performance", "quality", "quality_score")
# ``metric`` is the metric *name* on xRouteBench, not a score.
INPUT_FIELDS = ("input_tokens", "prompt_tokens")
OUTPUT_FIELDS = ("output_tokens", "completion_tokens")
COST_FIELDS = ("realized_cost", "cost_usd", "cost")
LATENCY_MS_FIELDS = ("latency_ms",)
LATENCY_S_FIELDS = ("response_time", "latency_s", "latency_sec", "latency_seconds", "latency")
ROUTER_SCORE_FIELDS = ("router_score", "router_win_rate", "p_strong")
ROUTER_ASSIGN_FIELDS = ("router_assignment", "routed_model")
PRICE_IN_FIELDS = ("input_price_per_1m", "input_usd_per_million")
PRICE_OUT_FIELDS = ("output_price_per_1m", "output_usd_per_million")
TOKEN_NUM_FIELDS = ("token_num", "total_tokens")

# Loaded from Hub; bulky text fields are dropped and never used for tokens/quality/cost.
KEEP_SOURCE_FIELDS = {
    str(a).strip().lower().replace(" ", "_")
    for group in (
        QUERY_ID_FIELDS,
        ("task_name",),
        MODEL_FIELDS,
        QUALITY_FIELDS,
        INPUT_FIELDS,
        OUTPUT_FIELDS,
        COST_FIELDS,
        LATENCY_MS_FIELDS,
        LATENCY_S_FIELDS,
        ROUTER_SCORE_FIELDS,
        ROUTER_ASSIGN_FIELDS,
        PRICE_IN_FIELDS,
        PRICE_OUT_FIELDS,
        TOKEN_NUM_FIELDS,
        ("size", "service", "api_model_id", "description", "query", "embedding_id"),
    )
    for a in group
}

InspectFn = Callable[[str, str, Path], tuple[list[str], list[str]]]
NamesFn = Callable[[str], Sequence[str]]


class XRouteBenchError(RuntimeError):
    """Loader / catalog failure (missing package, Hub error, empty panel)."""


@dataclass(frozen=True)
class CandidatePrices:
    """Published xRouteBench ``llm_candidates`` rates (USD per 1M tokens)."""

    model: str
    input_price_per_1m: float
    output_price_per_1m: float
    extra: dict[str, Any] = field(default_factory=dict)

    def inference_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            int(input_tokens) * self.input_price_per_1m / MILLION
            + int(output_tokens) * self.output_price_per_1m / MILLION
        )


def require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xRouteBench integration needs the Hugging Face `datasets` package. "
            "Install with: python3.13 -m pip install 'datasets>=3.0'. "
            "This is a dataset download, not a paid model API."
        ) from exc
    return __import__("datasets")


def cache_dir(path: Path | str | None = None) -> Path:
    dest = Path(path) if path is not None else CACHE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _feature_set(features: Sequence[str]) -> set[str]:
    return {_norm(f) for f in features}


def _first_key(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, Any]:
    lower = {_norm(k): (k, v) for k, v in row.items()}
    for alias in aliases:
        hit = lower.get(_norm(alias))
        if hit is None:
            continue
        key, val = hit
        if val is None or val == "":
            continue
        return str(key), val
    return None, None


def _has_any_column(columns: Sequence[str], aliases: Sequence[str]) -> bool:
    feats = _feature_set(columns)
    return any(_norm(a) in feats for a in aliases)


def classify_config(name: str, features: Sequence[str] | None = None) -> str:
    """Label a Hugging Face config from its *name and features*, not a fixed list."""
    n = _norm(name)
    feats = _feature_set(features or ())
    if _has_any_column(feats, PRICE_IN_FIELDS) and _has_any_column(feats, PRICE_OUT_FIELDS):
        return "pricing"
    if "candidate" in n:
        return "pricing"
    if "model_1" in feats and "model_2" in feats:
        return "pairwise"
    if n.endswith("_queries") or n.endswith("-queries"):
        if "model_name" not in feats and "model" not in feats:
            return "queries_only"
    if feats:
        if "model_name" in feats or "model" in feats:
            if any(k in feats for k in ("performance", "quality", "quality_score")):
                return "routing"
            return "routing_candidate"
        if ("query" in feats or "question" in feats) and "model_name" not in feats and "model" not in feats:
            return "queries_only"
        return "unknown"
    if n.endswith("_queries") or n.endswith("-queries"):
        return "queries_only"
    return "routing_candidate"


def inspect_hf_config(repo: str, name: str, cache: Path) -> tuple[list[str], list[str]]:
    datasets = require_datasets()
    builder = datasets.load_dataset_builder(repo, name, cache_dir=str(cache))
    info = builder.info
    features: list[str] = []
    if info is not None and info.features is not None:
        features = list(info.features.keys())
    splits: list[str] = []
    if info is not None and info.splits is not None:
        splits = list(info.splits.keys())
    if not splits:
        try:
            splits = list(datasets.get_dataset_split_names(repo, name))
        except Exception:
            splits = []
    return features, splits


def discover_catalog(
    repo: str = DEFAULT_REPO,
    *,
    cache: Path | None = None,
    get_names: NamesFn | None = None,
    inspect_config: InspectFn | None = None,
) -> list[dict[str, Any]]:
    """List Hub configs/splits and classify them. Names come from the Hub API."""
    dest = cache_dir(cache)
    if get_names is None:
        datasets = require_datasets()
        get_names = lambda r: list(datasets.get_dataset_config_names(r))
    inspect = inspect_config or inspect_hf_config
    names = list(get_names(repo))
    if not names:
        raise XRouteBenchError(f"no dataset configs reported for {repo!r}")
    catalog: list[dict[str, Any]] = []
    for name in names:
        try:
            features, splits = inspect(repo, name, dest)
        except Exception as exc:
            catalog.append(
                {
                    "name": name,
                    "kind": classify_config(name, None),
                    "features": [],
                    "splits": [],
                    "inspect_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        kind = classify_config(name, features)
        catalog.append(
            {
                "name": name,
                "kind": kind,
                "features": list(features),
                "splits": list(splits),
            }
        )
    return catalog


def routing_configs(catalog: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(c) for c in catalog if c.get("kind") in ("routing", "routing_candidate")]


def pricing_configs(catalog: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(c) for c in catalog if c.get("kind") == "pricing"]


def _records_from_dataset(ds: Any) -> list[dict[str, Any]]:
    if hasattr(ds, "to_list"):
        rows = ds.to_list()
        return [dict(r) for r in rows]
    return [dict(r) for r in ds]


def load_hf_split(
    repo: str,
    config: str,
    split: str,
    *,
    cache: Path | None = None,
) -> list[dict[str, Any]]:
    datasets = require_datasets()
    dest = cache_dir(cache)
    ds = datasets.load_dataset(repo, config, split=split, cache_dir=str(dest))
    if hasattr(ds, "column_names"):
        drop = [c for c in ds.column_names if _norm(c) not in KEEP_SOURCE_FIELDS]
        if drop:
            ds = ds.remove_columns(drop)
    return _records_from_dataset(ds)


def parse_price_table(records: Iterable[Mapping[str, Any]]) -> dict[str, CandidatePrices]:
    """USD/1M from the published candidate table. Rows with missing rates are dropped, not filled."""
    out: dict[str, CandidatePrices] = {}
    for raw in records:
        row = {str(k): v for k, v in raw.items()}
        _, name_v = _first_key(row, MODEL_FIELDS)
        name = as_str(name_v)
        _, pin_v = _first_key(row, PRICE_IN_FIELDS)
        _, pout_v = _first_key(row, PRICE_OUT_FIELDS)
        pin = as_float(pin_v)
        pout = as_float(pout_v)
        if not name or pin is None or pout is None:
            continue
        extra = {}
        for key in ("size", "service", "api_model_id", "description"):
            val = as_str(row.get(key))
            if val is not None:
                extra[key] = val
        out[name] = CandidatePrices(
            model=name,
            input_price_per_1m=pin,
            output_price_per_1m=pout,
            extra=extra,
        )
    return out


def load_price_table(
    repo: str = DEFAULT_REPO,
    *,
    catalog: Sequence[Mapping[str, Any]] | None = None,
    cache: Path | None = None,
    load_split: Callable[..., list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, CandidatePrices], dict[str, Any]]:
    """Load the detected pricing config. Empty table if none is present (costs stay missing)."""
    cat = list(catalog) if catalog is not None else discover_catalog(repo, cache=cache)
    pricing = pricing_configs(cat)
    meta: dict[str, Any] = {
        "available": False,
        "config": None,
        "split": None,
        "n_models": 0,
        "models": [],
        "note": "no pricing config detected; realized_cost stays missing",
    }
    if not pricing:
        return {}, meta
    cfg = pricing[0]
    splits = list(cfg.get("splits") or ("train",))
    split = splits[0]
    loader = load_split or load_hf_split
    records = loader(repo, cfg["name"], split, cache=cache)
    table = parse_price_table(records)
    meta.update(
        {
            "available": bool(table),
            "config": cfg["name"],
            "split": split,
            "n_models": len(table),
            "models": sorted(table),
            "n_source_rows": len(records),
            "note": (
                f"prices from Hugging Face config {cfg['name']!r} split {split!r}; "
                "EcoLogic models.json was not used"
            ),
        }
    )
    return table, meta


def detect_query_id_scheme(records: Sequence[Mapping[str, Any]]) -> str:
    """Choose an original-id scheme. Colliding ``task_id`` values become ``task_name::task_id``.

    ``embedding_id`` is never used. Rows with a missing ``task_id`` keep
    ``task_name::<original query text>``.
    """
    if not records:
        raise SchemaError("empty xRouteBench table")
    n = len(records)
    n_query_id = 0
    n_task_id = 0
    n_pair = 0
    n_task_or_query = 0
    by_task_id: dict[str, set[str | None]] = defaultdict(set)
    for raw in records:
        row = {str(k): v for k, v in raw.items()}
        if as_str(row.get("query_id")):
            n_query_id += 1
        tid = as_str(row.get("task_id"))
        tname = as_str(row.get("task_name"))
        query = as_str(row.get("query"))
        if tid:
            n_task_id += 1
            by_task_id[tid].add(tname)
        if tid and tname:
            n_pair += 1
        if tid or query:
            n_task_or_query += 1
    if n_query_id == n:
        return "query_id"
    if n_task_id == n and by_task_id and all(len(names) == 1 for names in by_task_id.values()):
        return "task_id"
    if n_pair == n:
        return "task_name::task_id"
    if n_task_or_query == n:
        return "task_name::task_id_or_query"
    raise SchemaError(
        "cannot preserve original query ids: need query_id, task_id, or query text "
        "(embedding_id is not used)"
    )


def query_id_from_row(row: Mapping[str, Any], scheme: str) -> str:
    if scheme == "query_id":
        qid = as_str(row.get("query_id"))
        if not qid:
            raise SchemaError("row missing original query_id")
        return qid
    tid = as_str(row.get("task_id"))
    tname = as_str(row.get("task_name"))
    if scheme == "task_id":
        if not tid:
            raise SchemaError("row missing original task_id")
        return tid
    if scheme == "task_name::task_id":
        if not tid or not tname:
            raise SchemaError("row missing original task_name/task_id")
        return f"{tname}::{tid}"
    if scheme == "task_name::task_id_or_query":
        if not tname:
            raise SchemaError("row missing original task_name")
        if tid:
            return f"{tname}::{tid}"
        query = as_str(row.get("query"))
        if not query:
            raise SchemaError("row missing original task_id and query text")
        return f"{tname}::{query}"
    raise SchemaError(f"unknown query id scheme {scheme!r}")


def _query_fingerprint(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    emb = row.get("embedding_id")
    return (
        as_str(row.get("task_name")),
        as_str(row.get("task_id")),
        as_str(emb) if emb not in (None, "") else None,
        as_str(row.get("query")),
    )


def assign_query_ids(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], str, list[str]]:
    """Original-field query ids. ``embedding_id`` is appended only when ``task_id``/query collides."""
    scheme = detect_query_id_scheme(records)
    notes: list[str] = []
    base = [query_id_from_row({str(k): v for k, v in raw.items()}, scheme) for raw in records]
    groups: dict[str, set[tuple[str | None, str | None, str | None, str | None]]] = defaultdict(set)
    for raw, qid in zip(records, base):
        groups[qid].add(_query_fingerprint({str(k): v for k, v in raw.items()}))
    ambiguous = {qid for qid, fps in groups.items() if len(fps) > 1}
    if not ambiguous:
        return base, scheme, notes
    refined: list[str] = []
    n_emb = 0
    n_query = 0
    for raw, qid in zip(records, base):
        if qid not in ambiguous:
            refined.append(qid)
            continue
        row = {str(k): v for k, v in raw.items()}
        emb = as_str(row.get("embedding_id")) if row.get("embedding_id") not in (None, "") else None
        if emb is not None:
            refined.append(f"{qid}::embedding_id={emb}")
            n_emb += 1
            continue
        query = as_str(row.get("query"))
        if query and query not in qid:
            refined.append(f"{qid}::{query}")
            n_query += 1
            continue
        refined.append(qid)
    notes.append(
        f"original id collisions on {len(ambiguous)} values; disambiguated with "
        f"published embedding_id ({n_emb} rows) and/or query text ({n_query} rows)"
    )
    return refined, f"{scheme}+disambiguate", notes


def _same_measurements(a: CanonicalRow, b: CanonicalRow) -> bool:
    return (
        a.quality == b.quality
        and a.input_tokens == b.input_tokens
        and a.output_tokens == b.output_tokens
        and a.realized_cost == b.realized_cost
        and a.latency_ms == b.latency_ms
    )


def collapse_duplicate_cells(rows: Sequence[CanonicalRow]) -> tuple[list[CanonicalRow], list[str]]:
    """Keep one row per (query_id, model). Identical measurements are dropped, not averaged."""
    notes: list[str] = []
    by: dict[tuple[str, str], CanonicalRow] = {}
    n_ident = 0
    conflicts: list[tuple[str, str]] = []
    for r in rows:
        key = (r.query_id, r.model)
        prev = by.get(key)
        if prev is None:
            by[key] = r
            continue
        if _same_measurements(prev, r):
            n_ident += 1
            continue
        conflicts.append(key)
    if n_ident:
        notes.append(f"dropped {n_ident} exact duplicate (query, model) rows; values were identical")
    if conflicts:
        raise SchemaError(
            f"conflicting measurements for {len(conflicts)} duplicate (query_id, model) cells; "
            f"example {conflicts[0]}"
        )
    return list(by.values()), notes


def _present(value: Any) -> bool:
    return value is not None and value != ""


def column_coverage(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, Any]:
    n = len(records)
    feats = set()
    for rec in records[:1]:
        feats.update(str(k) for k in rec.keys())
    # union of keys across a sample if the first row is sparse
    if records:
        for rec in records[: min(len(records), 50)]:
            feats.update(str(k) for k in rec.keys())
    out: dict[str, Any] = {}
    lower_map = {_norm(c): c for c in feats}
    for col in columns:
        src = lower_map.get(_norm(col))
        if src is None:
            out[col] = {
                "source_column": None,
                "column_present": False,
                "n_present": 0,
                "n_missing": n,
                "fraction_present": 0.0 if n else None,
            }
            continue
        n_present = sum(1 for rec in records if _present(rec.get(src)))
        out[col] = {
            "source_column": src,
            "column_present": True,
            "n_present": n_present,
            "n_missing": n - n_present,
            "fraction_present": (n_present / n) if n else None,
        }
    return out


def canonical_coverage(rows: Sequence[CanonicalRow]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {}
    for col in (*CANONICAL_COLUMNS, *OPTIONAL_COLUMNS):
        n_present = sum(1 for r in rows if getattr(r, col) is not None)
        out[col] = {
            "n_present": n_present,
            "n_missing": n - n_present,
            "fraction_present": (n_present / n) if n else None,
        }
    return out


def missing_field_report(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[CanonicalRow],
    *,
    prices: Mapping[str, CandidatePrices] | None = None,
) -> dict[str, Any]:
    """Mark source and canonical fields that are absent. Nothing is imputed here."""
    source_cols = (
        "query_id",
        "task_id",
        "task_name",
        "embedding_id",
        "model_name",
        "performance",
        "metric",
        "input_tokens",
        "output_tokens",
        "token_num",
        "response_time",
        "latency_ms",
        "cost",
        "realized_cost",
        *ROUTER_SCORE_FIELDS,
        *ROUTER_ASSIGN_FIELDS,
        *PRICE_IN_FIELDS,
        *PRICE_OUT_FIELDS,
    )
    priced = 0
    unpriced = 0
    if prices is not None:
        for r in rows:
            if r.model in prices:
                priced += 1
            else:
                unpriced += 1
    token_num_unused = column_coverage(records, TOKEN_NUM_FIELDS)
    return {
        "n_rows": len(rows),
        "source_columns": column_coverage(records, source_cols),
        "canonical": canonical_coverage(rows),
        "embedding_id_ignored": False,
        "embedding_id_not_used_as_primary_id": True,
        "token_num_not_used_for_cost": True,
        "token_num": token_num_unused,
        "models_with_published_price": priced,
        "models_without_published_price": unpriced,
        "ecologic_models_json_used": False,
    }


def map_routing_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    prices: Mapping[str, CandidatePrices] | None = None,
    query_id_scheme: str | None = None,
    source: str = "xroutebench",
) -> tuple[list[CanonicalRow], dict[str, Any]]:
    """Dedicated mapper. Does not call the generic alias table (avoids ``id``)."""
    if not records:
        raise SchemaError("empty xRouteBench routing table")
    ids, scheme, id_notes = assign_query_ids(records)
    prices = dict(prices or {})
    rows: list[CanonicalRow] = []
    n_cost_from_prices = 0
    n_cost_from_column = 0
    n_cost_missing = 0
    for raw, qid in zip(records, ids):
        row = {str(k): v for k, v in raw.items()}
        _, model_v = _first_key(row, MODEL_FIELDS)
        model = as_str(model_v)
        if not model:
            raise SchemaError(f"row {qid} has no model identity")
        _, qual_v = _first_key(row, QUALITY_FIELDS)
        quality = as_float(qual_v) if qual_v is not None else None
        _, in_v = _first_key(row, INPUT_FIELDS)
        _, out_v = _first_key(row, OUTPUT_FIELDS)
        in_tok = as_int(in_v) if in_v is not None else None
        out_tok = as_int(out_v) if out_v is not None else None
        _, cost_v = _first_key(row, COST_FIELDS)
        cost = as_float(cost_v) if cost_v is not None else None
        cost_source = "missing"
        if cost is not None:
            n_cost_from_column += 1
            cost_source = "metered"
        elif (
            in_tok is not None
            and out_tok is not None
            and model in prices
        ):
            cost = prices[model].inference_cost(in_tok, out_tok)
            n_cost_from_prices += 1
            cost_source = "price_table"
        else:
            n_cost_missing += 1
        lat_ms = None
        lat_col, lat_v = _first_key(row, LATENCY_MS_FIELDS)
        if lat_v is not None:
            lat_ms = as_float(lat_v)
        else:
            lat_col, lat_v = _first_key(row, LATENCY_S_FIELDS)
            if lat_v is not None:
                sec = as_float(lat_v)
                lat_ms = None if sec is None else sec * 1000.0
        _, score_v = _first_key(row, ROUTER_SCORE_FIELDS)
        _, assign_v = _first_key(row, ROUTER_ASSIGN_FIELDS)
        rows.append(
            CanonicalRow(
                query_id=qid,
                dataset=dataset,
                model=model,
                quality=quality,
                input_tokens=in_tok,
                output_tokens=out_tok,
                realized_cost=cost,
                latency_ms=lat_ms,
                router_score=as_float(score_v) if score_v is not None else None,
                router_assignment=as_str(assign_v),
                cost_source=cost_source,
            )
        )
    rows, drop_notes = collapse_duplicate_cells(rows)
    notes = {
        "query_id_scheme": scheme,
        "query_id_notes": id_notes + drop_notes,
        "n_cost_from_source_column": n_cost_from_column,
        "n_cost_from_published_prices": n_cost_from_prices,
        "n_cost_missing": n_cost_missing,
        "source": source,
        "latency_unit_note": (
            "response_time / latency fields in seconds are converted to ms; "
            "missing latency stays None (not 0)"
        ),
    }
    return rows, notes


def task_name_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    qids: dict[str, set[str]] = defaultdict(set)
    scheme = None
    try:
        scheme = detect_query_id_scheme(records)
    except SchemaError:
        scheme = None
    for raw in records:
        row = {str(k): v for k, v in raw.items()}
        name = as_str(row.get("task_name")) or "_missing_task_name"
        if scheme is not None:
            try:
                qids[name].add(query_id_from_row(row, scheme))
            except SchemaError:
                counts[name] += 1
        else:
            counts[name] += 1
    if qids:
        return {k: len(v) for k, v in sorted(qids.items())}
    return dict(sorted(counts.items()))


def filter_records_by_task(
    records: Sequence[Mapping[str, Any]],
    task_name: str,
) -> list[dict[str, Any]]:
    if task_name == "all":
        return [dict(r) for r in records]
    out = []
    for raw in records:
        row = {str(k): v for k, v in raw.items()}
        if as_str(row.get("task_name")) == task_name:
            out.append(row)
    return out


def _cell_ok(row: CanonicalRow, *, require_cost: bool) -> bool:
    if row.quality is None:
        return False
    if require_cost and row.realized_cost is None:
        return False
    return True


def select_complete_rows(
    rows: Sequence[CanonicalRow],
    *,
    require_cost: bool,
    min_queries: int = MIN_QUERIES,
    min_models: int = MIN_MODELS,
    max_models: int = MAX_MODELS,
) -> tuple[list[CanonicalRow], list[str]]:
    """Drop incomplete query×model cells. Never fills them."""
    notes: list[str] = []
    if not rows:
        return [], ["empty panel"]
    models = list(dict.fromkeys(r.model for r in rows))
    queries = list(dict.fromkeys(r.query_id for r in rows))
    covered: dict[str, set[str]] = {m: set() for m in models}
    for r in rows:
        if _cell_ok(r, require_cost=require_cost):
            covered[r.model].add(r.query_id)
    Q = set(queries)

    def pack(kept_models: Sequence[str], kept_queries: set[str], note: str) -> tuple[list[CanonicalRow], list[str]]:
        kept_m = list(kept_models)
        if len(kept_m) > max_models:
            kept_m = sorted(kept_m)[:max_models]
            notes.append(
                f"truncated to {max_models} models by name (MAX_MODELS); "
                "not ranked by coverage on the audited split"
            )
        mset, qset = set(kept_m), set(kept_queries)
        selected = [r for r in rows if r.model in mset and r.query_id in qset]
        # keep only complete cells among the selection
        selected = [r for r in selected if _cell_ok(r, require_cost=require_cost)]
        notes.append(note)
        return selected, notes

    full = [m for m in models if covered[m] == Q]
    if len(full) >= min_models and len(Q) >= min_queries:
        return pack(full, Q, f"complete-case: {len(full)} models on {len(Q)} queries")

    for frac, label in ((1.0, "100%"), (0.9, "90%"), (0.5, "50%")):
        good = [m for m in models if len(Q) > 0 and len(covered[m]) >= frac * len(Q)]
        if len(good) < min_models:
            continue
        inter = set.intersection(*(covered[m] for m in good)) if good else set()
        if len(inter) >= min_queries:
            return pack(
                good,
                inter,
                f"relaxed complete-case ({label} model coverage): "
                f"{len(good)} models on {len(inter)} of {len(Q)} queries",
            )
        notes.append(
            f"{label} coverage left {len(good)} models but only {len(inter)} complete queries"
        )

    return [], notes + [
        f"insufficient complete cells (need ≥{min_models} models and ≥{min_queries} queries; "
        f"no imputed quality/cost)"
    ]


def panel_from_rows(
    rows: Sequence[CanonicalRow],
    *,
    dataset: str,
    source: str,
    notes: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> LongPanel:
    check_model_count([r.model for r in rows])
    return LongPanel(
        rows=list(rows),
        dataset=dataset,
        source=source,
        notes=list(notes or []),
        extra=dict(extra or {}),
    )


def routing_panel_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    prices: Mapping[str, CandidatePrices] | None = None,
    source: str = "xroutebench",
    require_cost: bool = True,
    min_queries: int = MIN_QUERIES,
) -> tuple[LongPanel | None, dict[str, Any]]:
    """Map + complete-case. Returns ``None`` panel when the subset is not auditable."""
    mapped, map_notes = map_routing_records(
        records, dataset=dataset, prices=prices, source=source
    )
    coverage = missing_field_report(records, mapped, prices=prices)
    selected, sel_notes = select_complete_rows(
        mapped, require_cost=require_cost, min_queries=min_queries
    )
    info: dict[str, Any] = {
        "map": map_notes,
        "coverage": coverage,
        "selection_notes": sel_notes,
        "n_mapped_rows": len(mapped),
        "n_source_models": len({r.model for r in mapped}),
        "n_source_queries": len({r.query_id for r in mapped}),
        "source_models": sorted({r.model for r in mapped}),
        "task_name_query_counts": task_name_counts(records),
    }
    if not selected:
        info["status"] = "skipped"
        info["reason"] = sel_notes[-1] if sel_notes else "incomplete panel"
        return None, info
    try:
        panel = panel_from_rows(
            selected,
            dataset=dataset,
            source=source,
            notes=[
                f"query_id_scheme={map_notes['query_id_scheme']}",
                *sel_notes,
                "realized_cost from published xRouteBench prices × tokens only; "
                "EcoLogic models.json was not used",
                "token_num is never split or used as a token substitute",
            ],
            extra=info,
        )
    except SchemaError as exc:
        info["status"] = "skipped"
        info["reason"] = str(exc)
        return None, info
    info["status"] = "ok"
    info["n_panel_rows"] = panel.n_rows
    info["n_panel_models"] = panel.n_models
    info["n_panel_queries"] = panel.n_queries
    return panel, info
