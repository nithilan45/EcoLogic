"""Stage 1 - build a training/calibration pool disjoint from the frozen test set.

GSM8K : 400 items whose test-split indices are NOT in the frozen 100.
MMLU  : 400 items from the 47 subjects NOT used by the frozen selection.
MBPP  : 400 items. HumanEval is entirely inside the frozen test set, so it is
        not reused for training under any framing; MBPP substitutes for it.

Overlap against the frozen test set is asserted programmatically on item IDs and
on the underlying dataset indices, and the PASS/FAIL result is printed.

Pool is split 70/30 into TRAIN and CALIBRATION. The frozen 364-item test set is
never touched here.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from build_benchmark import LETTERS, fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "router_v2"
FROZEN = ROOT / "raw_results" / "benchmark_items.json"

SEED = 771113  # distinct from the frozen set's seed (20260905)
N_GSM8K = 400
N_MMLU = 400
N_MBPP = 400
TRAIN_FRAC = 0.70

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def load_frozen() -> dict:
    with open(FROZEN) as f:
        d = json.load(f)
    man = d["selection_manifest"]
    return {
        "item_ids": {it["item_id"] for it in d["items"]},
        "gsm8k_indices": set(man["gsm8k"]["indices"]),
        "mmlu_subjects": set(man["mmlu"]["subjects"]),
        "mmlu_subject_indices": {s: set(v) for s, v in man["mmlu"]["indices"].items()},
    }


def build_gsm8k(rng, frozen) -> list[dict]:
    path = fetch("gsm8k_test.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    available = [i for i in range(len(rows)) if i not in frozen["gsm8k_indices"]]
    idxs = sorted(rng.sample(available, N_GSM8K))
    items = []
    for i in idxs:
        r = rows[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "item_id": f"gsm8k/{i}",
            "benchmark": "gsm8k",
            "raw_query": r["question"],
            "prompt": (
                "Solve this math word problem. Show your work, then give the final "
                'numeric answer on the last line in the form "Answer: N".\n\n'
                f"{r['question']}"
            ),
            "grading": "numeric",
            "reference": gold,
            "source_index": i,
        })
    return items


def build_mmlu(rng, frozen) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(fetch("mmlu_test.parquet")).to_pylist()
    by_subject: dict[str, list[dict]] = {}
    for k, r in enumerate(rows):
        by_subject.setdefault(r["subject"], []).append((k, r))

    # Use only subjects the frozen selection did not touch. This makes overlap
    # impossible by construction, not just by index checking.
    fresh = sorted(s for s in by_subject if s not in frozen["mmlu_subjects"])
    per = {}
    remaining = N_MMLU
    for n, s in enumerate(fresh):
        take = min(len(by_subject[s]), remaining // (len(fresh) - n))
        per[s] = take
        remaining -= take
    items = []
    manifest = {}
    for s, take in per.items():
        if take == 0:
            continue
        pool = by_subject[s]
        picks = sorted(rng.sample(range(len(pool)), take))
        manifest[s] = picks
        for p in picks:
            gidx, r = pool[p]
            choices = list(r["choices"])
            if len(choices) != 4:
                continue
            lettered = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
            items.append({
                "item_id": f"mmlu/{s}/{p}",
                "benchmark": "mmlu",
                "subject": s,
                "raw_query": f"{r['question']}\n{lettered}",
                "prompt": (
                    "Answer this multiple-choice question. Respond with only the "
                    "letter of the correct option (A, B, C, or D) on the last line "
                    'in the form "Answer: X".\n\n'
                    f"{r['question']}\n{lettered}"
                ),
                "grading": "mcq",
                "reference": LETTERS[int(r["answer"])],
                "choices": choices,
                "source_index": p,
            })
    return items, {"subjects_used": sorted(manifest), "indices": manifest}


def build_mbpp(rng) -> list[dict]:
    import urllib.request
    cache = Path("/tmp/bench_cache/mbpp.jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(MBPP_URL, cache)
    rows = [json.loads(l) for l in open(cache) if l.strip()]
    picks = sorted(rng.sample(range(len(rows)), N_MBPP))
    items = []
    for p in picks:
        r = rows[p]
        tests = "\n".join(r["test_list"])
        items.append({
            "item_id": f"mbpp/{r['task_id']}",
            "benchmark": "mbpp",
            "raw_query": r["text"],
            "prompt": (
                "Write a Python function for the following task. Return the "
                "complete implementation inside a single ```python code block, "
                "with no tests and no explanation.\n\n"
                f"Task: {r['text']}\n\n"
                f"Your function must satisfy these assertions:\n{tests}"
            ),
            "grading": "execute_mbpp",
            "reference": None,
            "mbpp_tests": r["test_list"],
            "mbpp_setup": r.get("test_setup_code") or "",
            "source_index": r["task_id"],
        })
    return items


def overlap_check(items: list[dict], frozen: dict) -> bool:
    ids = [it["item_id"] for it in items]
    dup_ids = sorted(set(ids) & frozen["item_ids"])

    gsm_idx = {it["source_index"] for it in items if it["benchmark"] == "gsm8k"}
    gsm_overlap = sorted(gsm_idx & frozen["gsm8k_indices"])

    mmlu_subj = {it["subject"] for it in items if it["benchmark"] == "mmlu"}
    subj_overlap = sorted(mmlu_subj & frozen["mmlu_subjects"])

    he = [i for i in ids if i.startswith("HumanEval/")]

    print("=" * 62)
    print("OVERLAP CHECK vs FROZEN 364-ITEM TEST SET")
    print("=" * 62)
    print(f"  pool items                          : {len(ids)} ({len(set(ids))} unique)")
    print(f"  shared item_ids with frozen set     : {len(dup_ids)} -> {dup_ids[:5]}")
    print(f"  shared GSM8K test-split indices     : {len(gsm_overlap)} -> {gsm_overlap[:5]}")
    print(f"  MMLU subjects shared with frozen    : {len(subj_overlap)} -> {subj_overlap[:5]}")
    print(f"  HumanEval items reused for training : {len(he)}")
    ok = not dup_ids and not gsm_overlap and not subj_overlap and not he and len(set(ids)) == len(ids)
    print(f"\n  RESULT: {'PASS - pool is disjoint from the frozen test set' if ok else 'FAIL'}")
    print("=" * 62)
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen()
    rng = random.Random(SEED)

    gsm = build_gsm8k(rng, frozen)
    mmlu, mmlu_man = build_mmlu(rng, frozen)
    mbpp = build_mbpp(rng)
    items = gsm + mmlu + mbpp

    ok = overlap_check(items, frozen)
    if not ok:
        raise SystemExit("ABORT: pool overlaps the frozen test set; refusing to continue.")

    rng.shuffle(items)
    n_train = int(round(TRAIN_FRAC * len(items)))
    train, calib = items[:n_train], items[n_train:]

    meta = {
        "seed": SEED,
        "frozen_test_set": str(FROZEN.relative_to(ROOT)),
        "overlap_check": "PASS",
        "counts": {
            "gsm8k": len(gsm), "mmlu": len(mmlu), "mbpp": len(mbpp), "total": len(items),
        },
        "mmlu_manifest": mmlu_man,
        "gsm8k_indices": sorted(it["source_index"] for it in gsm),
        "mbpp_task_ids": sorted(it["source_index"] for it in mbpp),
        "split": {"train_frac": TRAIN_FRAC, "n_train": len(train), "n_calibration": len(calib)},
    }
    for name, subset in (("train_pool.json", train), ("calibration_pool.json", calib)):
        with open(OUT / name, "w") as f:
            json.dump({**meta, "split_name": name.split("_")[0], "items": subset}, f, indent=2)
        print(f"wrote router_v2/{name}: {len(subset)} items")

    import collections
    for label, subset in (("TRAIN", train), ("CALIBRATION", calib)):
        c = collections.Counter(it["benchmark"] for it in subset)
        print(f"  {label:<12} {dict(c)}")


if __name__ == "__main__":
    main()
