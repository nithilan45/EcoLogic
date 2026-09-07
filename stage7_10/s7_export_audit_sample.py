"""Export a reviewable sample of Stage 7 raw generations.

Stages 1-6 committed their full raw responses (`raw_results/graded.jsonl`,
9 MB). Stage 7's are 288 MB raw and ~48 MB gzipped, because Tier 1 averages
~3,100 output tokens per call, so committing them in full would burden a repo
that is also deployed. The per-sample tables (`s7_*_samples.csv.gz`) already
carry every grade, token count and cost the reports derive from, but they do not
carry response text, so they cannot be used to check whether a given item was
*graded* correctly.

This exports the response text for the cases where grading is most likely to be
wrong or contested, so that judgement is auditable at a few MB instead of 48:

  - every ungradable and every truncated response (rare, and the places a grader
    is most likely to be wrong)
  - a seeded sample of items flagged `no_tier_correct` (all three tiers failed:
    the label-noise cases), with all 9 responses each
  - a seeded sample of items whose k=3 samples split 2-1, carrying only the
    disagreeing tier's 3 samples, since that is the judgement the majority vote
    resolves
  - a seeded random sample of ordinary responses per (tier, benchmark) as a
    control, so the export is not composed only of hard cases

Caps keep this a few MB rather than the ~48 MB a full export would take; the
counts per category are printed and recorded so the sampling is not silent.

Full raw text remains on disk at stage7_10/s7_{pool,test}_{responses,graded}.jsonl
and is regenerable via s7_run.py + s7_grade.py, at ~$41 and roughly 15 hours.
Because the provider models carry no immutable revision, a regeneration would
not be identical.

Usage: python3 stage7_10/s7_export_audit_sample.py
"""

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEED = 20260906
CONTROL_PER_CELL = 8
MAX_NO_TIER_CORRECT_ITEMS = 40
MAX_DISAGREEMENT_ITEMS_PER_TIER = 40


def load_flagged(target: str) -> tuple[set[str], dict[str, set[int]]]:
    """Return (no_tier_correct item_ids, {item_id: tiers that split 2-1})."""
    path = OUT / f"s7_{target}_labels.json"
    if not path.exists():
        return set(), {}
    lab = json.loads(path.read_text())["labels"]
    noisy = {i for i, d in lab.items() if d.get("no_tier_correct")}
    split = {}
    for i, d in lab.items():
        tiers = {int(t) for t, u in (d.get("unanimous_by_tier") or {}).items() if not u}
        if tiers:
            split[i] = tiers
    return noisy, split


def export(target: str) -> None:
    src = OUT / f"s7_{target}_graded.jsonl"
    if not src.exists():
        print(f"{target}: no graded file, skipped")
        return

    rng = random.Random(SEED)
    noisy, split = load_flagged(target)

    keep_noisy = set(rng.sample(sorted(noisy),
                                min(MAX_NO_TIER_CORRECT_ITEMS, len(noisy))))
    # sample disagreement items per disagreeing tier, keeping only that tier
    by_tier = defaultdict(list)
    for i, tiers in split.items():
        for t in tiers:
            by_tier[t].append(i)
    keep_split = defaultdict(set)
    for t, ids in sorted(by_tier.items()):
        for i in rng.sample(sorted(ids),
                            min(MAX_DISAGREEMENT_ITEMS_PER_TIER, len(ids))):
            keep_split[i].add(t)

    rows = []
    control_pool = defaultdict(list)
    for line in open(src):
        if not line.strip():
            continue
        r = json.loads(line)
        reason = None
        if not r.get("gradable", True):
            reason = "ungradable"
        elif r.get("truncated"):
            reason = "truncated"
        elif r["item_id"] in keep_noisy:
            reason = "no_tier_correct"
        elif r["tier"] in keep_split.get(r["item_id"], ()):
            reason = "k3_disagreement"
        if reason:
            r["_audit_reason"] = reason
            rows.append(r)
        elif r["item_id"] not in noisy and r["item_id"] not in split:
            control_pool[(r["tier"], r["benchmark"])].append(r)

    for cell, pool in sorted(control_pool.items()):
        for r in rng.sample(pool, min(CONTROL_PER_CELL, len(pool))):
            r["_audit_reason"] = "control_sample"
            rows.append(r)

    rows.sort(key=lambda r: (r["item_id"], r["tier"], r["sample_idx"]))
    dst = OUT / f"s7_{target}_audit_sample.jsonl.gz"
    with gzip.open(dst, "wt", compresslevel=9) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts = defaultdict(int)
    for r in rows:
        counts[r["_audit_reason"]] += 1
    mb = dst.stat().st_size / 1e6
    print(f"{target}: {len(rows)} responses -> {dst.name} ({mb:.2f} MB)")
    for k, v in sorted(counts.items()):
        print(f"    {k:<18} {v}")


if __name__ == "__main__":
    for t in ("pool", "test"):
        export(t)
