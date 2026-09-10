"""Read frozen experimental outputs. Refuse any write into hashed trees."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from woais_experiments.paths import (
    HASH_MANIFEST,
    RESULTS,
    ROOT,
    assert_inside_results,
    is_frozen,
)

WRITE_MODES = set("wa+x")


class FrozenTreeError(RuntimeError):
    """Attempted mutation of an immutable experimental artifact."""


@dataclass(frozen=True)
class HashRecord:
    sha256: str
    size: int
    relpath: str

    @property
    def path(self) -> Path:
        return ROOT / self.relpath


def parse_manifest(text: str | None = None) -> list[HashRecord]:
    raw = HASH_MANIFEST.read_text() if text is None else text
    records: list[HashRecord] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rest = line.split(None, 1)
        size_s, relpath = rest.split(None, 1)
        records.append(HashRecord(digest, int(size_s), relpath.strip()))
    return records


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_hashes(records: Iterable[HashRecord] | None = None) -> dict[str, Any]:
    recs = list(records) if records is not None else parse_manifest()
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    for rec in recs:
        if not rec.path.exists():
            missing.append(rec.relpath)
            continue
        digest = sha256_file(rec.path)
        size = rec.path.stat().st_size
        if digest != rec.sha256 or size != rec.size:
            mismatches.append({
                "path": rec.relpath,
                "expected_sha256": rec.sha256,
                "actual_sha256": digest,
                "expected_bytes": rec.size,
                "actual_bytes": size,
            })
    ok = not mismatches and not missing
    return {
        "ok": ok,
        "n_records": len(recs),
        "n_mismatches": len(mismatches),
        "n_missing": len(missing),
        "mismatches": mismatches,
        "missing": missing,
    }


def open_frozen(path: Path, mode: str = "r", **kwargs):
    if any(c in mode for c in WRITE_MODES):
        raise FrozenTreeError(f"refusing write mode {mode!r} on {path}")
    if not is_frozen(path) and path.resolve() != HASH_MANIFEST.resolve():
        # Allow reading non-frozen files too, but never writing them via this helper.
        pass
    return open(path, mode, **kwargs)


def write_result(relpath: str | Path, data: Any, *, json_indent: int = 2) -> Path:
    dest = RESULTS / relpath if not isinstance(relpath, Path) else relpath
    dest = assert_inside_results(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if dest.suffix == ".json" or str(dest).endswith(".json"):
        tmp.write_text(json.dumps(data, indent=json_indent) + "\n")
    elif isinstance(data, str):
        tmp.write_text(data if data.endswith("\n") else data + "\n")
    elif isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(json.dumps(data, indent=json_indent) + "\n")
    tmp.replace(dest)
    return dest


def write_result_bytes(relpath: str, payload: bytes) -> Path:
    dest = assert_inside_results(RESULTS / relpath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------ loaders

NEEDED_CALL_FIELDS = (
    "item_id", "tier", "model", "benchmark", "subject",
    "correct", "gradable", "truncated",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "usd", "latency_s", "retries", "finish_reason",
)


def _as_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _normalize_call(row: dict) -> dict:
    return {
        "item_id": str(row.get("item_id")),
        "tier": _as_int(row.get("tier")),
        "model": row.get("model"),
        "benchmark": row.get("benchmark"),
        "subject": row.get("subject"),
        "correct": _as_bool(row.get("correct")),
        "gradable": _as_bool(row.get("gradable")),
        "truncated": _as_bool(row.get("truncated")),
        "prompt_tokens": _as_int(row.get("prompt_tokens")) or 0,
        "completion_tokens": _as_int(row.get("completion_tokens")) or 0,
        "total_tokens": _as_int(row.get("total_tokens")) or 0,
        "usd": _as_float(row.get("usd")) or 0.0,
        "latency_s": _as_float(row.get("latency_s")),
        "retries": _as_int(row.get("retries")) or 0,
        "finish_reason": row.get("finish_reason"),
        "sample_idx": _as_int(row.get("sample_idx")),
        "temperature": _as_float(row.get("temperature")),
    }


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_stage12_calls() -> list[dict]:
    path = ROOT / "raw_results" / "graded.jsonl"
    return [_normalize_call(r) for r in iter_jsonl(path)]


def load_stage12_routing() -> dict:
    with open_frozen(ROOT / "raw_results" / "routing.json") as fh:
        return json.load(fh)


def load_json(path: Path) -> Any:
    with open_frozen(path) as fh:
        return json.load(fh)


def load_s7_samples(which: str = "test") -> list[dict]:
    path = ROOT / "stage7_10" / f"s7_{which}_samples.csv.gz"
    rows: list[dict] = []
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(_normalize_call(row))
    return rows


@dataclass
class ItemMatrix:
    """Per-item, per-tier measurements for a complete three-tier set."""

    item_ids: list[str]
    bench_of: dict[str, str]
    correct: dict[tuple[int, str], bool]
    tokens: dict[tuple[int, str], int]
    prompt_tokens: dict[tuple[int, str], int]
    completion_tokens: dict[tuple[int, str], int]
    usd: dict[tuple[int, str], float]
    latency_s: dict[tuple[int, str], float]
    model_of: dict[int, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.item_ids)


def matrix_from_calls(calls: list[dict], tiers: tuple[int, ...] = (1, 2, 3)) -> ItemMatrix:
    correct, tokens, usd, lat = {}, {}, {}, {}
    pt, ct = {}, {}
    bench_of: dict[str, str] = {}
    model_of: dict[int, str] = {}
    for r in calls:
        t, i = r["tier"], r["item_id"]
        if t is None or i is None:
            continue
        key = (int(t), str(i))
        correct[key] = bool(r["correct"])
        tokens[key] = int(r["total_tokens"] or 0)
        pt[key] = int(r["prompt_tokens"] or 0)
        ct[key] = int(r["completion_tokens"] or 0)
        usd[key] = float(r["usd"] or 0.0)
        lat[key] = float(r["latency_s"] or 0.0)
        if r.get("benchmark"):
            bench_of[str(i)] = r["benchmark"]
        if r.get("model"):
            model_of[int(t)] = r["model"]
    item_ids = sorted({i for (_, i) in correct}, key=lambda x: (bench_of.get(x, ""), x))
    complete = [i for i in item_ids if all((t, i) in correct for t in tiers)]
    return ItemMatrix(complete, bench_of, correct, tokens, pt, ct, usd, lat, model_of)


def load_stage12_matrix() -> ItemMatrix:
    return matrix_from_calls(load_stage12_calls())


def load_s7_test_matrix(sample_idx: int = 0) -> ItemMatrix:
    rows = [r for r in load_s7_samples("test") if r.get("sample_idx") == sample_idx]
    return matrix_from_calls(rows)
