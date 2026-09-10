"""Build and freeze the held-out train/val/test split.

Dataset
-------
Largest EcoLogic panel with query text, per-tier correctness, tokens, and
USD: ``router_v2/pool_graded.jsonl.gz`` (1200 items × 3 tiers). It is disjoint
from the frozen Stage 1–2 n=364 panel (``raw_results/benchmark_items.json``).
That 364-query result is not reused, resplit, or reinterpreted here.

Grouping
--------
MMLU items share a subject → one group per subject (no subject in two splits).
GSM8K and MBPP have no smaller family than the item, so each query is its own
group. Prompt template is collinear with benchmark (three templates) and is
recorded, not used as the grouping key.

Splits are 60/20/20 **within each benchmark**, so every split contains GSM8K,
MBPP, and MMLU. Exact normalized duplicates, and token-Jaccard ≥ 0.9 pairs,
are merged into the same group before allocation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.frozen import ItemMatrix, matrix_from_calls, sha256_file
from woais_experiments.paths import PACKAGE, ROOT
from woais_experiments.routing.ablations import subset_matrix

HELDOUT_DIR = PACKAGE / "heldout"
MANIFEST_PATH = HELDOUT_DIR / "split_manifest.json"
SPLIT_SEED = 20260910
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
JACCARD_NEAR = 0.90
POOL_GRADED = ROOT / "router_v2" / "pool_graded.jsonl.gz"
TRAIN_POOL = ROOT / "router_v2" / "train_pool.json"
CAL_POOL = ROOT / "router_v2" / "calibration_pool.json"
STAGE12_ITEMS = ROOT / "raw_results" / "benchmark_items.json"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


class SplitError(RuntimeError):
    """Held-out split protocol was violated."""


def normalize_text(text: str) -> str:
    s = _PUNCT.sub(" ", (text or "").lower())
    return _WS.sub(" ", s).strip()


def token_set(text: str) -> frozenset[str]:
    return frozenset(normalize_text(text).split())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_item_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in (TRAIN_POOL, CAL_POOL):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for it in payload["items"]:
            qid = str(it["item_id"])
            out[qid] = {
                "item_id": qid,
                "benchmark": str(it.get("benchmark") or "unknown"),
                "subject": str(it.get("subject") or "") or None,
                "raw_query": str(it.get("raw_query") or ""),
                "prompt": str(it.get("prompt") or ""),
                "source_index": it.get("source_index"),
                "legacy_split": str(it.get("split") or payload.get("split_name") or ""),
            }
    return out


def load_graded_calls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(POOL_GRADED, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_panel() -> tuple[ItemMatrix, dict[str, dict[str, Any]]]:
    records = load_item_records()
    matrix = matrix_from_calls(load_graded_calls())
    missing = [i for i in matrix.item_ids if i not in records]
    if missing:
        raise SplitError(f"graded items missing query text: {missing[:8]}")
    extra = set(records) - set(matrix.item_ids)
    if extra:
        raise SplitError(f"query records without a complete 3-tier grade: {len(extra)}")
    s12 = json.loads(STAGE12_ITEMS.read_text(encoding="utf-8"))
    s12_ids = {str(it["item_id"]) for it in s12["items"]}
    overlap = set(matrix.item_ids) & s12_ids
    if overlap:
        raise SplitError(
            f"refusing to mix Stage 1–2 n=364 items into the held-out pool "
            f"({len(overlap)} overlapping ids)"
        )
    return matrix, records


def natural_group_id(rec: Mapping[str, Any]) -> str:
    bench = str(rec["benchmark"])
    subj = rec.get("subject")
    if bench == "mmlu" and subj:
        return f"mmlu/{subj}"
    return f"{bench}/item/{rec['item_id']}"


def prompt_template_id(rec: Mapping[str, Any]) -> str:
    return str(rec["benchmark"])


def merge_duplicate_groups(
    records: Mapping[str, Mapping[str, Any]],
    base_groups: Mapping[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Collapse exact and near-duplicate queries into one group.

    Returns (item_id → group_id, flags).
    """
    ids = sorted(records)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    flags: list[dict[str, Any]] = []
    norm = {i: normalize_text(records[i]["raw_query"]) for i in ids}
    tokens = {i: token_set(records[i]["raw_query"]) for i in ids}

    by_norm: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        if norm[i]:
            by_norm[norm[i]].append(i)
    for text, members in by_norm.items():
        if len(members) < 2:
            continue
        for b in members[1:]:
            union(members[0], b)
        flags.append({
            "kind": "exact_normalized",
            "item_ids": members,
            "text_sha256": _sha256_text(text),
        })

    # Near-duplicates: O(n²) is fine at n=1200.
    for i in range(len(ids)):
        a = ids[i]
        ta = tokens[a]
        if len(ta) < 4:
            continue
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if find(a) == find(b):
                continue
            sim = jaccard(ta, tokens[b])
            if sim >= JACCARD_NEAR:
                union(a, b)
                flags.append({
                    "kind": "token_jaccard",
                    "item_ids": [a, b],
                    "jaccard": round(float(sim), 6),
                    "threshold": JACCARD_NEAR,
                })

    # Keep natural groups intact: union everyone already sharing a base group.
    by_base: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_base[base_groups[i]].append(i)
    for members in by_base.values():
        for b in members[1:]:
            union(members[0], b)

    item_group = {}
    cluster_rep: dict[str, str] = {}
    for i in ids:
        root = find(i)
        if root not in cluster_rep:
            # Prefer an mmlu subject label if any member has one.
            members = [j for j in ids if find(j) == root]
            mmlu = [j for j in members if records[j]["benchmark"] == "mmlu" and records[j].get("subject")]
            if mmlu:
                cluster_rep[root] = f"mmlu/{records[mmlu[0]]['subject']}|dup:{root}"
            else:
                cluster_rep[root] = f"cluster:{root}"
        item_group[i] = cluster_rep[root]
    return item_group, flags


def allocate_stratified_grouped(
    item_ids: Sequence[str],
    group_of: Mapping[str, str],
    bench_of: Mapping[str, str],
    *,
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, str]:
    """Assign each group to train/val/test. Stratified by benchmark.

    Allocation is by item count, not group count, so singleton GSM8K/MBPP
    groups and 8–9 item MMLU subjects can coexist.
    """
    rng = np.random.default_rng(int(seed))
    groups: dict[str, list[str]] = defaultdict(list)
    for i in item_ids:
        groups[group_of[i]].append(i)

    def group_stratum(gid: str) -> str:
        counts = Counter(bench_of[i] for i in groups[gid])
        return sorted(counts, key=lambda b: (-counts[b], b))[0]

    item_split: dict[str, str] = {}
    for bench in sorted(set(group_stratum(g) for g in groups)):
        gids = [g for g in groups if group_stratum(g) == bench]
        gids = [gids[i] for i in rng.permutation(len(gids))]
        n = sum(len(groups[g]) for g in gids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise SplitError(f"stratum {bench!r} is too small for 60/20/20 (n={n})")
        counts = {"train": 0, "val": 0, "test": 0}
        targets = {"train": n_train, "val": n_val, "test": n_test}
        for gid in gids:
            remaining = {s: targets[s] - counts[s] for s in targets}
            pick = max(remaining, key=lambda s: (remaining[s], {"train": 2, "val": 1, "test": 0}[s]))
            for i in groups[gid]:
                item_split[i] = pick
            counts[pick] += len(groups[gid])
    missing = [i for i in item_ids if i not in item_split]
    if missing:
        raise SplitError("split missed items")
    return item_split


def _assert_disjoint(split: Mapping[str, Sequence[str]]) -> None:
    t, v, te = set(split["train"]), set(split["val"]), set(split["test"])
    if t & v or t & te or v & te:
        raise SplitError("train/val/test are not disjoint")
    if not t or not v or not te:
        raise SplitError("empty split")


def cross_split_overlap_flags(
    records: Mapping[str, Mapping[str, Any]],
    item_split: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Any remaining exact/near duplicate that still straddles splits."""
    by_split: dict[str, list[str]] = defaultdict(list)
    for qid, sp in item_split.items():
        by_split[sp].append(qid)
    flags: list[dict[str, Any]] = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for ia in by_split[a]:
            na = normalize_text(records[ia]["raw_query"])
            ta = token_set(records[ia]["raw_query"])
            for ib in by_split[b]:
                nb = normalize_text(records[ib]["raw_query"])
                if na and na == nb:
                    flags.append({
                        "kind": "exact_normalized_cross_split",
                        "splits": [a, b],
                        "item_ids": [ia, ib],
                    })
                    continue
                sim = jaccard(ta, token_set(records[ib]["raw_query"]))
                if sim >= JACCARD_NEAR:
                    flags.append({
                        "kind": "token_jaccard_cross_split",
                        "splits": [a, b],
                        "item_ids": [ia, ib],
                        "jaccard": round(float(sim), 6),
                    })
    return flags


def build_manifest(*, seed: int = SPLIT_SEED, force: bool = False) -> dict[str, Any]:
    if MANIFEST_PATH.exists() and not force:
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return existing

    matrix, records = load_panel()
    ids = list(matrix.item_ids)
    bench_of = {i: records[i]["benchmark"] for i in ids}
    base = {i: natural_group_id(records[i]) for i in ids}
    group_of, dup_flags = merge_duplicate_groups(records, base)
    item_split = allocate_stratified_grouped(ids, group_of, bench_of, seed=seed)
    split = {
        "train": sorted(i for i in ids if item_split[i] == "train"),
        "val": sorted(i for i in ids if item_split[i] == "val"),
        "test": sorted(i for i in ids if item_split[i] == "test"),
    }
    _assert_disjoint(split)
    if set(split["train"]) | set(split["val"]) | set(split["test"]) != set(ids):
        raise SplitError("split does not cover the panel")

    leftover = cross_split_overlap_flags(records, item_split)
    if leftover:
        raise SplitError(
            f"near-duplicate queries still cross splits after merging "
            f"({len(leftover)} pairs). Refusing to freeze."
        )

    queries = []
    for qid in ids:
        rec = records[qid]
        raw = rec["raw_query"]
        queries.append({
            "item_id": qid,
            "split": item_split[qid],
            "group_id": group_of[qid],
            "benchmark": rec["benchmark"],
            "subject": rec.get("subject"),
            "prompt_template": prompt_template_id(rec),
            "source_index": rec.get("source_index"),
            "text_sha256": _sha256_text(normalize_text(raw)),
            "raw_query_sha256": _sha256_text(raw),
        })

    body = {
        "protocol": "heldout_v1",
        "note": (
            "Stage 1–2 n=364 (raw_results/) is historical and is not this split. "
            "This panel is router_v2/pool_graded.jsonl.gz, verified disjoint from Stage 1–2."
        ),
        "seed": int(seed),
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1.0 - TRAIN_FRAC - VAL_FRAC, 4)},
        "n": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "total": len(ids),
        },
        "n_by_benchmark": {
            sp: {
                b: sum(1 for i in split[sp] if bench_of[i] == b)
                for b in sorted(set(bench_of.values()))
            }
            for sp in ("train", "val", "test")
        },
        "dataset": {
            "graded": "router_v2/pool_graded.jsonl.gz",
            "queries": ["router_v2/train_pool.json", "router_v2/calibration_pool.json"],
            "graded_sha256": sha256_file(POOL_GRADED),
            "train_pool_sha256": sha256_file(TRAIN_POOL),
            "calibration_pool_sha256": sha256_file(CAL_POOL),
        },
        "grouping": {
            "mmlu": "subject",
            "gsm8k": "item (singleton)",
            "mbpp": "item (singleton)",
            "near_duplicate_jaccard": JACCARD_NEAR,
            "embeddings": "not used (no precomputed embeddings loaded)",
        },
        "duplicate_clusters_merged": dup_flags,
        "n_duplicate_clusters_merged": len(dup_flags),
        "cross_split_overlap": leftover,
        "split_ids": split,
        "queries": queries,
    }
    body["manifest_sha256"] = _sha256_text(_canonical({k: v for k, v in body.items() if k != "manifest_sha256"}))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"split manifest missing: {p}. Run build_split.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def split_ids(manifest: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    m = manifest or load_manifest()
    return {k: list(m["split_ids"][k]) for k in ("train", "val", "test")}


def subset(matrix: ItemMatrix, ids: Sequence[str]) -> ItemMatrix:
    return subset_matrix(matrix, ids)


def verify_manifest_hash(manifest: Mapping[str, Any] | None = None) -> str:
    m = dict(manifest or load_manifest())
    stored = m.get("manifest_sha256")
    recomputed = _sha256_text(_canonical({k: v for k, v in m.items() if k != "manifest_sha256"}))
    if stored != recomputed:
        raise SplitError("split_manifest.json hash mismatch — file was edited after freeze")
    return recomputed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Freeze the held-out EcoLogic split.")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--force", action="store_true", help="Overwrite an existing split_manifest.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    existed = MANIFEST_PATH.exists()
    man = build_manifest(seed=int(args.seed), force=bool(args.force))
    n = man["n"]
    print(f"TRAIN n\t{n['train']}")
    print(f"VAL n\t{n['val']}")
    print(f"TEST n\t{n['test']}")
    print(f"manifest\t{MANIFEST_PATH}")
    print(f"wrote\t{not existed or bool(args.force)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
