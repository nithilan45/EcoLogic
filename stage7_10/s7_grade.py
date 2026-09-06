"""Stage 7 - grade k=3 samples per item per tier and build majority-vote labels.

Grading is identical in kind to Stages 1-6 (imported from benchmark/grade.py and
router_v2/grade_pool.py so it cannot drift): MMLU by letter match, GSM8K by
numeric final-answer match, MBPP by executing the official assertions in a
resource-limited subprocess.

The pre-registered label rule:
  - per-tier grade  = MAJORITY over the k=3 samples (correct iff >= 2 of 3)
  - per-tier energy = MEAN total tokens over the k=3 samples, at the paper's rates
  - oracle label    = lowest-energy tier whose majority grade is correct;
                      if no tier qualifies, the lowest-energy tier, flagged as
                      label noise rather than ground truth

Only items with all 3 tiers x all 3 samples present are labelled. Incomplete
items are excluded entirely (never partially imputed) and counted, which
implements the pre-registered throughput contingency.

Usage: python3 stage7_10/s7_grade.py --target pool|test
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "router_v2"))
from api import MODELS  # noqa: E402
from grade import (extract_code, extract_letter, extract_number,  # noqa: E402
                   _norm_num)
from grade_pool import mbpp_entry_point, run_mbpp_one  # noqa: E402

OUT = ROOT / "stage7_10"
TIERS = [1, 2, 3]
K = 3
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}

TARGETS = {
    "pool": {"items": ["s7_train_pool.json", "s7_calibration_pool.json"],
             "resp": "s7_pool_responses.jsonl",
             "graded": "s7_pool_graded.jsonl", "labels": "s7_pool_labels.json"},
    "test": {"items": ["s7_test_set.json"],
             "resp": "s7_test_responses.jsonl",
             "graded": "s7_test_graded.jsonl", "labels": "s7_test_labels.json"},
}


def load_items(target: str) -> dict:
    items = {}
    for name in TARGETS[target]["items"]:
        with open(OUT / name) as f:
            d = json.load(f)
        for it in d["items"]:
            items[it["item_id"]] = {**it, "split": d["split_name"]}
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    args = ap.parse_args()
    cfg = TARGETS[args.target]

    items = load_items(args.target)
    rows, seen = [], set()
    with open(OUT / cfg["resp"]) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("answer"):
                continue
            key = (r["tier"], r["item_id"], r["sample_idx"])
            if key in seen or r["item_id"] not in items:
                continue
            seen.add(key)
            rows.append(r)
    print(f"{len(rows)} successful responses loaded")

    jobs, idx = [], []
    for i, r in enumerate(rows):
        it = items[r["item_id"]]
        text = r.get("answer") or ""
        r["truncated"] = r.get("finish_reason") == "length"
        if it["benchmark"] == "mbpp":
            code = extract_code(text, mbpp_entry_point(it["mbpp_tests"]))
            r["extracted_code"] = code
            jobs.append((code, it["mbpp_setup"], it["mbpp_tests"]))
            idx.append(i)
        elif it["benchmark"] == "mmlu":
            pred = extract_letter(text)
            r["extracted"] = pred
            r["correct"] = pred is not None and pred == it["reference"]
            r["gradable"] = pred is not None
        else:
            pred = extract_number(text)
            gold = _norm_num(it["reference"])
            r["extracted"] = pred
            r["correct"] = pred is not None and gold is not None and abs(pred - gold) < 1e-4
            r["gradable"] = pred is not None

    if jobs:
        print(f"executing {len(jobs)} MBPP candidates ...", flush=True)
        with ProcessPoolExecutor(max_workers=8) as ex:
            for i, res in zip(idx, ex.map(run_mbpp_one, jobs, chunksize=8)):
                rows[i]["correct"] = res["correct"]
                rows[i]["exec_status"] = res["exec_status"]
                rows[i]["gradable"] = res["exec_status"] != "no_code"

    with open(OUT / cfg["graded"], "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote stage7_10/{cfg['graded']} ({len(rows)} rows)")

    # ---------------- majority vote per (tier, item)
    by = defaultdict(list)
    for r in rows:
        by[(r["tier"], r["item_id"])].append(r)

    labels, n_noisy, flip = {}, 0, Counter()
    complete = [i for i in items
                if all(len(by.get((t, i), [])) >= K for t in TIERS)]
    for i in complete:
        maj, energy, unan = {}, {}, {}
        for t in TIERS:
            rs = sorted(by[(t, i)], key=lambda x: x["sample_idx"])[:K]
            n_ok = sum(bool(x.get("correct")) for x in rs)
            maj[t] = n_ok >= 2
            unan[t] = n_ok in (0, K)
            mean_tok = sum(int(x.get("total_tokens") or 0) for x in rs) / K
            energy[t] = mean_tok / 1000 * PAPER_RATES[t]
            flip[(t, n_ok)] += 1
        winners = [t for t in TIERS if maj[t]]
        if winners:
            lab, noisy = min(winners, key=lambda t: energy[t]), False
        else:
            lab, noisy = min(TIERS, key=lambda t: energy[t]), True
            n_noisy += 1
        labels[i] = {
            "oracle_tier": lab, "no_tier_correct": noisy,
            "benchmark": items[i]["benchmark"], "split": items[i]["split"],
            "majority_correct_by_tier": {str(t): maj[t] for t in TIERS},
            "unanimous_by_tier": {str(t): unan[t] for t in TIERS},
            "energy_by_tier_J": {str(t): round(energy[t], 4) for t in TIERS},
            "mean_tokens_by_tier": {
                str(t): round(sum(int(x.get("total_tokens") or 0)
                                  for x in sorted(by[(t, i)],
                                                  key=lambda x: x["sample_idx"])[:K]) / K, 1)
                for t in TIERS},
        }

    incomplete = sorted(set(items) - set(complete))
    n_split = Counter(v["split"] for v in labels.values())
    # how often did the 3 samples disagree? this is the label noise the majority
    # vote is designed to absorb, measured rather than assumed
    disagree = {str(t): {"unanimous": flip[(t, 0)] + flip[(t, K)],
                         "split_2_1": flip[(t, 1)] + flip[(t, 2)]} for t in TIERS}
    payload = {
        "k": K, "rule": "majority of k=3; energy = mean tokens over k=3",
        "n_items_requested": len(items), "n_items_labelled": len(labels),
        "n_incomplete_excluded": len(incomplete),
        "incomplete_item_ids": incomplete[:50],
        "n_no_tier_correct_flagged": n_noisy,
        "labelled_by_split": dict(n_split),
        "sample_disagreement_by_tier": disagree,
        "label_distribution": dict(Counter(v["oracle_tier"] for v in labels.values())),
        "label_distribution_excluding_noisy": dict(
            Counter(v["oracle_tier"] for v in labels.values() if not v["no_tier_correct"])),
        "labels": labels,
    }
    with open(OUT / cfg["labels"], "w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote stage7_10/{cfg['labels']}")
    print(f"  requested / labelled     : {len(items)} / {len(labels)}")
    print(f"  excluded (incomplete k=3): {len(incomplete)}")
    print(f"  flagged no-tier-correct  : {n_noisy}")
    print(f"  labelled by split        : {dict(n_split)}")
    print(f"  oracle label mix         : {payload['label_distribution']}")
    print("  k=3 sample disagreement (2-1 splits, i.e. the noise majority voting removes):")
    for t in TIERS:
        d = disagree[str(t)]
        tot = d["unanimous"] + d["split_2_1"]
        if tot:
            print(f"    tier {t}: {d['split_2_1']}/{tot} = {d['split_2_1'] / tot:.1%} of items")

    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[(r["tier"], r["benchmark"])]
        a[0] += int(bool(r.get("correct")))
        a[1] += 1
    print("\nper-sample accuracy (all k samples pooled):")
    for k in sorted(agg):
        c, n = agg[k]
        print(f"  tier {k[0]} {k[1]:<6} {c}/{n} = {c / n:.1%}")


if __name__ == "__main__":
    main()
