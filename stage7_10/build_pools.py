"""Stage 7 - build a fresh training pool and a NEW frozen test set.

Disjointness is asserted against three prior selections and printed PASS/FAIL:
  1. the original 364-item frozen test set (raw_results/benchmark_items.json)
  2. Stage 1's 1,200-item pool (router_v2/{train,calibration}_pool.json)
  3. the new frozen test set vs everything, including the new pool

GSM8K pool items come from the GSM8K **train** split; every prior GSM8K
selection used only the test split, so overlap there is impossible by
construction. MBPP has 974 problems total and Stage 1 consumed 400, so after
reserving 164 for the new test set exactly 410 remain for the pool - the code
portion cannot be scaled. That ceiling is pre-registered, not discovered here.

The pool is shuffled before writing so that any prefix of generated items is a
uniform random subsample (the throughput contingency in the pre-registration).
"""

import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from build_benchmark import LETTERS, fetch  # noqa: E402

OUT = ROOT / "stage7_10"
CACHE = Path("/tmp/bench_cache")

SEED = 20260906
N_POOL_TARGET = 5000
N_MBPP_TEST = 164
N_MMLU_TEST = 100
N_GSM8K_TEST = 100
N_MBPP_POOL = 410
TRAIN_FRAC = 0.70

GSM8K_TRAIN_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
                   "master/grade_school_math/data/train.jsonl")


def prior_ids() -> dict:
    with open(ROOT / "raw_results" / "benchmark_items.json") as f:
        frozen = json.load(f)
    frozen_ids = {it["item_id"] for it in frozen["items"]}
    s1 = set()
    for name in ("train_pool.json", "calibration_pool.json"):
        with open(ROOT / "router_v2" / name) as f:
            for it in json.load(f)["items"]:
                s1.add(it["item_id"])
    return {"frozen": frozen_ids, "stage1": s1,
            "frozen_mmlu_subject_idx": {
                s: set(v) for s, v in frozen["selection_manifest"]["mmlu"]["indices"].items()}}


def mmlu_used_subject_idx() -> set:
    """(subject, index) pairs already consumed, parsed from prior item ids."""
    used = set()
    for path in (ROOT / "raw_results" / "benchmark_items.json",
                 ROOT / "router_v2" / "train_pool.json",
                 ROOT / "router_v2" / "calibration_pool.json"):
        with open(path) as f:
            for it in json.load(f)["items"]:
                if it["item_id"].startswith("mmlu/"):
                    _, subj, idx = it["item_id"].split("/")
                    used.add((subj, int(idx)))
    return used


def gsm8k_item(i, r, split_tag):
    gold = r["answer"].split("####")[-1].strip().replace(",", "")
    return {
        "item_id": f"gsm8k-{split_tag}/{i}",
        "benchmark": "gsm8k",
        "raw_query": r["question"],
        "prompt": ("Solve this math word problem. Show your work, then give the final "
                   'numeric answer on the last line in the form "Answer: N".\n\n'
                   f"{r['question']}"),
        "grading": "numeric", "reference": gold, "source_index": i,
        "source_split": split_tag,
    }


def mmlu_item(subj, idx, r):
    choices = list(r["choices"])
    lettered = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
    return {
        "item_id": f"mmlu/{subj}/{idx}",
        "benchmark": "mmlu", "subject": subj,
        "raw_query": f"{r['question']}\n{lettered}",
        "prompt": ("Answer this multiple-choice question. Respond with only the letter "
                   "of the correct option (A, B, C, or D) on the last line in the form "
                   '"Answer: X".\n\n'
                   f"{r['question']}\n{lettered}"),
        "grading": "mcq", "reference": LETTERS[int(r["answer"])], "choices": choices,
        "source_index": idx,
    }


def mbpp_item(r):
    tests = "\n".join(r["test_list"])
    return {
        "item_id": f"mbpp/{r['task_id']}",
        "benchmark": "mbpp",
        "raw_query": r["text"],
        "prompt": ("Write a Python function for the following task. Return the complete "
                   "implementation inside a single ```python code block, with no tests "
                   "and no explanation.\n\n"
                   f"Task: {r['text']}\n\n"
                   f"Your function must satisfy these assertions:\n{tests}"),
        "grading": "execute_mbpp", "reference": None,
        "mbpp_tests": r["test_list"], "mbpp_setup": r.get("test_setup_code") or "",
        "source_index": r["task_id"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    prior = prior_ids()
    banned = prior["frozen"] | prior["stage1"]
    mmlu_used = mmlu_used_subject_idx()

    # ---------------- MBPP: hard ceiling, reserve test items first
    mbpp_rows = [json.loads(l) for l in open(fetch_mbpp()) if l.strip()]
    mbpp_avail = [r for r in mbpp_rows if f"mbpp/{r['task_id']}" not in banned]
    rng.shuffle(mbpp_avail)
    mbpp_test = [mbpp_item(r) for r in mbpp_avail[:N_MBPP_TEST]]
    mbpp_pool = [mbpp_item(r) for r in mbpp_avail[N_MBPP_TEST:N_MBPP_TEST + N_MBPP_POOL]]

    # ---------------- GSM8K: pool from TRAIN split, test from unused TEST split
    gtrain = [json.loads(l) for l in open(fetch_gsm8k_train()) if l.strip()]
    gtest = [json.loads(l) for l in open(fetch("gsm8k_test.jsonl")) if l.strip()]
    test_avail = [i for i in range(len(gtest)) if f"gsm8k/{i}" not in banned]
    gsm_test_idx = sorted(rng.sample(test_avail, N_GSM8K_TEST))
    gsm8k_test = [gsm8k_item(i, gtest[i], "test") for i in gsm_test_idx]

    # ---------------- MMLU: disjoint (subject, index) pairs
    import pyarrow.parquet as pq
    mmlu_rows = pq.read_table(fetch("mmlu_test.parquet")).to_pylist()
    by_subject = {}
    for r in mmlu_rows:
        by_subject.setdefault(r["subject"], []).append(r)
    mmlu_candidates = []
    for subj in sorted(by_subject):
        for idx, r in enumerate(by_subject[subj]):
            if (subj, idx) not in mmlu_used and len(r["choices"]) == 4:
                mmlu_candidates.append((subj, idx, r))
    rng.shuffle(mmlu_candidates)
    mmlu_test = [mmlu_item(s, i, r) for s, i, r in mmlu_candidates[:N_MMLU_TEST]]

    # ---------------- fill the pool to target with GSM8K-train + MMLU
    n_remaining = N_POOL_TARGET - len(mbpp_pool)
    n_each = n_remaining // 2
    gsm_pool_idx = sorted(rng.sample(range(len(gtrain)), n_each))
    gsm8k_pool = [gsm8k_item(i, gtrain[i], "train") for i in gsm_pool_idx]
    mmlu_pool = [mmlu_item(s, i, r)
                 for s, i, r in mmlu_candidates[N_MMLU_TEST:N_MMLU_TEST + (n_remaining - n_each)]]

    pool = mbpp_pool + gsm8k_pool + mmlu_pool
    test = mbpp_test + mmlu_test + gsm8k_test

    # ---------------- disjointness assertions
    pool_ids = [it["item_id"] for it in pool]
    test_ids = [it["item_id"] for it in test]
    checks = [
        ("pool vs original frozen test set", set(pool_ids) & prior["frozen"]),
        ("pool vs Stage 1 pool", set(pool_ids) & prior["stage1"]),
        ("new test set vs original frozen test set", set(test_ids) & prior["frozen"]),
        ("new test set vs Stage 1 pool", set(test_ids) & prior["stage1"]),
        ("new test set vs Stage 7 pool", set(test_ids) & set(pool_ids)),
        ("pool internal duplicates", {i for i in pool_ids if pool_ids.count(i) > 1} if len(set(pool_ids)) != len(pool_ids) else set()),
        ("test internal duplicates", {i for i in test_ids if test_ids.count(i) > 1} if len(set(test_ids)) != len(test_ids) else set()),
    ]
    print("=" * 68)
    print("STAGE 7 DISJOINTNESS CHECK")
    print("=" * 68)
    ok = True
    for label, bad in checks:
        print(f"  {label:<46} {len(bad):>4} shared  {sorted(bad)[:3]}")
        ok = ok and not bad
    he = [i for i in pool_ids + test_ids if i.startswith("HumanEval/")]
    print(f"  {'HumanEval items reused anywhere':<46} {len(he):>4}")
    ok = ok and not he
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    if not ok:
        raise SystemExit("ABORT: Stage 7 selections are not disjoint; refusing to continue.")

    rng.shuffle(pool)  # prefix of generated items == random subsample
    n_train = int(round(TRAIN_FRAC * len(pool)))
    meta = {
        "seed": SEED,
        "pool_target": N_POOL_TARGET,
        "disjointness_check": "PASS",
        "k_samples": 3, "pool_temperature": 0.7, "test_temperature": 0.0,
        "counts": {
            "pool_total": len(pool),
            "pool_mbpp": len(mbpp_pool), "pool_gsm8k": len(gsm8k_pool),
            "pool_mmlu": len(mmlu_pool),
            "test_total": len(test), "test_mbpp": len(mbpp_test),
            "test_mmlu": len(mmlu_test), "test_gsm8k": len(gsm8k_test),
        },
        "mbpp_ceiling_note": ("MBPP has 974 problems total; Stage 1 used 400 and 164 are "
                              "reserved for the new test set, leaving exactly 410 for the "
                              "pool. The code portion cannot be scaled."),
    }
    for name, subset in (("s7_train_pool.json", pool[:n_train]),
                         ("s7_calibration_pool.json", pool[n_train:]),
                         ("s7_test_set.json", test)):
        with open(OUT / name, "w") as f:
            json.dump({**meta, "split_name": name, "items": subset}, f, indent=2)
        print(f"wrote stage7_10/{name}: {len(subset)} items")
    print(f"\ncounts: {meta['counts']}")
    print(f"scale-up vs Stage 1 pool: {len(pool) / 1200:.2f}x overall, "
          f"MBPP {len(mbpp_pool) / 400:.2f}x, GSM8K {len(gsm8k_pool) / 400:.2f}x, "
          f"MMLU {len(mmlu_pool) / 400:.2f}x")


def fetch_mbpp() -> Path:
    p = CACHE / "mbpp.jsonl"
    if not p.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google-research/google-research/"
            "master/mbpp/mbpp.jsonl", p)
    return p


def fetch_gsm8k_train() -> Path:
    p = CACHE / "gsm8k_train.jsonl"
    if not p.exists():
        urllib.request.urlretrieve(GSM8K_TRAIN_URL, p)
    return p


if __name__ == "__main__":
    main()
