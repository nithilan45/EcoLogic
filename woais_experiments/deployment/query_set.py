"""Freeze the deployment-real query subset BEFORE any inference.

IDs and text hashes are stored first. Execution may use a prefix of this list
for dry-run validation; the frozen set itself is not retuned on outcomes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.frozen import load_stage12_matrix
from woais_experiments.routing.ablations import load_item_texts

QUERY_SET_SEED = 20260910
TARGET_N = 80
MIN_N = 50
MAX_N = 150


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_query_set(items: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in sorted(items, key=lambda r: str(r["query_id"])):
        h.update(str(row["query_id"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(row["text_sha256"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _stratified_sample(
    ids: Sequence[str],
    bench_of: Mapping[str, str],
    *,
    n: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(int(seed))
    by_b: dict[str, list[str]] = {}
    for qid in ids:
        by_b.setdefault(str(bench_of.get(qid, "unknown")), []).append(str(qid))
    for bench in by_b:
        rng.shuffle(by_b[bench])
    chosen: list[str] = []
    benches = sorted(by_b)
    if not benches:
        return []
    # Round-robin so GSM8K / MMLU / MBPP stay represented.
    while len(chosen) < n:
        progressed = False
        for bench in benches:
            bucket = by_b[bench]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return sorted(chosen)


def select_query_set(
    *,
    n: int = TARGET_N,
    seed: int = QUERY_SET_SEED,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Select a fixed query subset. Does not look at deployment quality/cost."""
    n_req = int(n)
    if not allow_small and (n_req < MIN_N or n_req > MAX_N):
        raise ValueError(f"official query-set size must be in [{MIN_N}, {MAX_N}], got {n_req}")
    texts = load_item_texts()
    matrix = load_stage12_matrix()
    eligible = [
        qid
        for qid in matrix.item_ids
        if qid in texts and str(texts[qid].get("raw_query") or "").strip()
    ]
    if not eligible:
        raise RuntimeError("no Stage 1–2 items with raw_query text")
    take = min(n_req, len(eligible))
    ids = _stratified_sample(eligible, matrix.bench_of, n=take, seed=int(seed))
    items = []
    for qid in ids:
        text = str(texts[qid]["raw_query"])
        items.append({
            "query_id": qid,
            "benchmark": str(matrix.bench_of.get(qid) or texts[qid].get("benchmark") or "unknown"),
            "text": text,
            "text_sha256": _sha256_text(text),
            "n_chars": len(text),
        })
    payload = {
        "protocol": "deployment_real_v2",
        "frozen_before_benchmark": True,
        "seed": int(seed),
        "n_requested": n_req,
        "n": len(items),
        "query_ids": [r["query_id"] for r in items],
        "items": items,
        "query_set_hash": hash_query_set(items),
        "source": "raw_results/benchmark_items.json ∩ Stage 1–2 complete-case matrix",
        "note": (
            "Query IDs were selected before any deployment-real inference. "
            "The same IDs are used for every policy."
        ),
    }
    payload["query_ids_sha256"] = _sha256_text("\n".join(payload["query_ids"]))
    return payload


def executed_slice(query_set: Mapping[str, Any], *, max_queries: int | None) -> list[dict[str, Any]]:
    items = list(query_set["items"])
    if max_queries is None:
        return items
    n = max(1, int(max_queries))
    return items[: min(n, len(items))]


def as_prompts(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"id": str(r["query_id"]), "text": str(r["text"])} for r in items]
