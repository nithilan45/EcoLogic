"""Read frozen experimental outputs. Refuse any write into hashed trees."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from woais_experiments.paths import (
    HASH_MANIFEST,
    RESULTS,
    ROOT,
    assert_inside_results,
    get_artifact_hook,
    get_overwrite_policy,
    get_results_root,
    get_run_meta,
    is_frozen,
    looks_like_home_absolute,
    public_relpath,
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


class ResultExistsError(RuntimeError):
    """Refusing to clobber an existing artifact (runner overwrite policy)."""


def to_jsonable(obj: Any) -> Any:
    """JSON-safe values. Non-finite floats become strings, never silent null.

    Filesystem paths inside the repo are stored repo-relative. Home-directory
    and temp-directory absolutes are redacted so artifacts cannot deanonymize
    an author via ``/Users/<name>``.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return public_relpath(obj)
    if isinstance(obj, str):
        if looks_like_home_absolute(obj):
            return public_relpath(obj)
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    return obj


def _destination(relpath: str | Path) -> Path:
    if isinstance(relpath, Path) and relpath.is_absolute():
        return relpath
    return get_results_root() / relpath


def _stamp_payload(data: Any) -> Any:
    meta = get_run_meta()
    if not meta or not isinstance(data, dict) or "woais_run" in data:
        return data
    return {**data, "woais_run": dict(meta)}


def _respect_existing(dest: Path) -> Path | None:
    """Return dest to keep, or None to write. Raises if overwrite is forbidden."""
    if not dest.exists():
        return None
    policy = get_overwrite_policy()
    if policy == "replace":
        return None
    if policy == "resume":
        hook = get_artifact_hook()
        if hook is not None:
            hook(dest, True)
        return dest
    if policy == "force":
        logging.getLogger("woais").warning("overwriting existing artifact %s", dest)
        return None
    raise ResultExistsError(
        f"refusing to overwrite {dest}; pass --resume or --force on run_woais.py"
    )


def write_result(
    relpath: str | Path,
    data: Any,
    *,
    json_indent: int = 2,
    clobber: bool = False,
) -> Path:
    dest = assert_inside_results(_destination(relpath))
    if not clobber:
        kept = _respect_existing(dest)
        if kept is not None:
            return kept
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _stamp_payload(data)
    tmp = dest.with_name(dest.name + ".tmp")
    if dest.suffix == ".json" or str(dest).endswith(".json"):
        payload = to_jsonable(payload)
        tmp.write_text(
            json.dumps(payload, indent=json_indent, default=str, allow_nan=False) + "\n"
        )
    elif isinstance(payload, str):
        tmp.write_text(payload if payload.endswith("\n") else payload + "\n")
    elif isinstance(payload, bytes):
        tmp.write_bytes(payload)
    else:
        payload = to_jsonable(payload)
        tmp.write_text(
            json.dumps(payload, indent=json_indent, default=str, allow_nan=False) + "\n"
        )
    tmp.replace(dest)
    hook = get_artifact_hook()
    if hook is not None:
        hook(dest, False)
    return dest


def write_result_bytes(relpath: str, payload: bytes) -> Path:
    dest = assert_inside_results(_destination(relpath))
    kept = _respect_existing(dest)
    if kept is not None:
        return kept
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    hook = get_artifact_hook()
    if hook is not None:
        hook(dest, False)
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
        "prompt_tokens": _as_int(row.get("prompt_tokens")),
        "completion_tokens": _as_int(row.get("completion_tokens")),
        "total_tokens": _as_int(row.get("total_tokens")),
        "usd": _as_float(row.get("usd")),
        "latency_s": _as_float(row.get("latency_s")),
        "retries": _as_int(row.get("retries")),
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
    n_dropped_incomplete: int = 0

    @property
    def n(self) -> int:
        return len(self.item_ids)


def _call_is_complete(row: dict) -> bool:
    """Refuse to invent $0 / 0s / incorrect for missing measurements."""
    if row.get("tier") is None or row.get("item_id") is None:
        return False
    if row.get("correct") is None:
        return False
    for key in (
        "usd",
        "latency_s",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        if row.get(key) is None:
            return False
    return True


def matrix_from_calls(calls: list[dict], tiers: tuple[int, ...] = (1, 2, 3)) -> ItemMatrix:
    correct, tokens, usd, lat = {}, {}, {}, {}
    pt, ct = {}, {}
    bench_of: dict[str, str] = {}
    model_of: dict[int, str] = {}
    n_dropped = 0
    for r in calls:
        t, i = r["tier"], r["item_id"]
        if t is None or i is None:
            n_dropped += 1
            continue
        if not _call_is_complete(r):
            n_dropped += 1
            continue
        key = (int(t), str(i))
        correct[key] = bool(r["correct"])
        tokens[key] = int(r["total_tokens"])
        pt[key] = int(r["prompt_tokens"])
        ct[key] = int(r["completion_tokens"])
        usd[key] = float(r["usd"])
        lat[key] = float(r["latency_s"])
        if r.get("benchmark"):
            bench_of[str(i)] = r["benchmark"]
        if r.get("model"):
            model_of[int(t)] = r["model"]
    item_ids = sorted({i for (_, i) in correct}, key=lambda x: (bench_of.get(x, ""), x))
    complete = [i for i in item_ids if all((t, i) in correct for t in tiers)]
    if n_dropped:
        logging.getLogger("woais").warning(
            "dropped %s incomplete calls (missing usd/latency/tokens/correct; not zero-filled)",
            n_dropped,
        )
    return ItemMatrix(
        complete, bench_of, correct, tokens, pt, ct, usd, lat, model_of,
        n_dropped_incomplete=n_dropped,
    )


def load_stage12_matrix() -> ItemMatrix:
    return matrix_from_calls(load_stage12_calls())


def load_s7_test_matrix(sample_idx: int = 0) -> ItemMatrix:
    rows = [r for r in load_s7_samples("test") if r.get("sample_idx") == sample_idx]
    return matrix_from_calls(rows)
