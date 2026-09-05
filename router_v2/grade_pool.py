"""Stage 1 (cont.) - grade the pool and compute oracle labels.

Grading is identical in kind to the frozen-test-set run: MMLU by letter match,
GSM8K by numeric final-answer match, code by executing the official test
assertions in a resource-limited subprocess. MBPP replaces HumanEval as the
pool's code benchmark, so an MBPP executor is added here; the extraction and
sandbox settings are imported from benchmark/grade.py so they cannot drift.

Oracle label = lowest-energy tier that answered the item correctly. Items where
no tier was correct are labelled with the lowest-energy tier (the convention
used in the previous report) and flagged as label noise, not ground truth.

Writes router_v2/pool_graded.jsonl and router_v2/pool_labels.json.
"""

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "router_v2"))
from grade import (EXEC_TIMEOUT, SANDBOX_PREAMBLE, extract_code,  # noqa: E402
                   extract_letter, extract_number, _norm_num)
from api import MODELS  # noqa: E402

OUT = ROOT / "router_v2"
TIERS = [1, 2, 3]
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}


def mbpp_entry_point(tests: list[str]) -> str:
    for t in tests:
        m = re.search(r"assert\s+\(?\s*([A-Za-z_]\w*)\s*\(", t)
        if m:
            return m.group(1)
    return ""


def run_mbpp_one(args) -> dict:
    code, setup, tests = args
    if not code.strip():
        return {"correct": False, "exec_status": "no_code", "stderr": ""}
    program = (
        SANDBOX_PREAMBLE
        + "\n" + code
        + "\n" + (setup or "")
        + "\n" + "\n".join(tests)
        + "\nprint('__OK__')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "cand.py"
        f.write_text(program)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", str(f)],
                               capture_output=True, text=True,
                               timeout=EXEC_TIMEOUT + 5, cwd=td)
        except subprocess.TimeoutExpired:
            return {"correct": False, "exec_status": "timeout", "stderr": ""}
        ok = p.returncode == 0 and "__OK__" in p.stdout
        status = "pass" if ok else ("assert_or_error" if p.returncode else "no_ok_marker")
        return {"correct": ok, "exec_status": status, "stderr": (p.stderr or "")[-600:]}


def load_items() -> dict:
    items = {}
    for name in ("train_pool.json", "calibration_pool.json"):
        with open(OUT / name) as f:
            d = json.load(f)
        for it in d["items"]:
            items[it["item_id"]] = {**it, "split": d["split_name"]}
    return items


def main():
    items = load_items()
    rows, seen = [], set()
    with open(OUT / "pool_responses.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["tier"], r["item_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    jobs, idx = [], []
    for i, r in enumerate(rows):
        it = items[r["item_id"]]
        text = r.get("answer") or ""
        r["truncated"] = r.get("finish_reason") == "length"
        r["split"] = it["split"]

        if it["benchmark"] == "mbpp":
            ep = mbpp_entry_point(it["mbpp_tests"])
            code = extract_code(text, ep)
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
            for i, res in zip(idx, ex.map(run_mbpp_one, jobs)):
                rows[i]["correct"] = res["correct"]
                rows[i]["exec_status"] = res["exec_status"]
                rows[i]["exec_stderr"] = res["stderr"]
                rows[i]["gradable"] = res["exec_status"] != "no_code"

    with open(OUT / "pool_graded.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote router_v2/pool_graded.jsonl ({len(rows)} rows)")

    # ---- oracle labels
    correct, tokens = {}, {}
    for r in rows:
        correct[(r["tier"], r["item_id"])] = bool(r.get("correct"))
        tokens[(r["tier"], r["item_id"])] = int(r.get("total_tokens") or 0)

    labels = {}
    complete = [i for i in items
                if all((t, i) in correct for t in TIERS)]
    n_noisy = 0
    for i in complete:
        e = {t: tokens[(t, i)] / 1000 * PAPER_RATES[t] for t in TIERS}
        winners = [t for t in TIERS if correct[(t, i)]]
        if winners:
            lab = min(winners, key=lambda t: e[t])
            noisy = False
        else:
            lab = min(TIERS, key=lambda t: e[t])
            noisy = True
            n_noisy += 1
        labels[i] = {
            "oracle_tier": lab,
            "no_tier_correct": noisy,
            "benchmark": items[i]["benchmark"],
            "split": items[i]["split"],
            "correct_by_tier": {str(t): correct[(t, i)] for t in TIERS},
            "energy_by_tier_J": {str(t): round(e[t], 4) for t in TIERS},
            "tokens_by_tier": {str(t): tokens[(t, i)] for t in TIERS},
        }

    incomplete = sorted(set(items) - set(complete))
    payload = {
        "n_items": len(labels),
        "n_incomplete_excluded": len(incomplete),
        "incomplete_item_ids": incomplete[:50],
        "n_no_tier_correct_flagged": n_noisy,
        "label_distribution": dict(Counter(v["oracle_tier"] for v in labels.values())),
        "label_distribution_excluding_noisy": dict(
            Counter(v["oracle_tier"] for v in labels.values() if not v["no_tier_correct"])),
        "labels": labels,
    }
    with open(OUT / "pool_labels.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote router_v2/pool_labels.json")
    print(f"  labelled items          : {len(labels)}")
    print(f"  excluded (missing tier) : {len(incomplete)}")
    print(f"  flagged no-tier-correct : {n_noisy}")
    print(f"  oracle label mix        : {payload['label_distribution']}")

    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[(r["tier"], r["benchmark"])]
        a[0] += int(bool(r.get("correct")))
        a[1] += 1
    print("\nper-tier accuracy on the pool:")
    for k in sorted(agg):
        c, n = agg[k]
        print(f"  tier {k[0]} {k[1]:<6} {c}/{n} = {c / n:.1%}")


if __name__ == "__main__":
    main()
