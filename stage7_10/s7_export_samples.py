"""Export the compact per-sample table that every Stage 7 number derives from.

The graded JSONL carries full response text: 45,000 pool samples come to ~288 MB
raw and still ~50 MB gzipped, because Tier 1 averages ~3,150 output tokens per
call. Earlier stages kept raw generations out of git for the same reason
(`raw_results/*.parquet` is gitignored), so this follows that precedent and
commits the grades, token counts, cost and latency instead — ~0.9 MB, and
sufficient to recompute every accuracy, energy and cost figure in the report.

Raw text stays on disk at stage7_10/s7_{pool,test}_{responses,graded}.jsonl and
is regenerable from s7_run.py + s7_grade.py.

Usage: python3 stage7_10/s7_export_samples.py
"""

import csv
import gzip
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

COLS = ["tier", "model", "item_id", "sample_idx", "benchmark", "subject", "split",
        "temperature", "latency_s", "retries", "prompt_tokens", "completion_tokens",
        "total_tokens", "usd", "finish_reason", "truncated", "extracted", "correct",
        "gradable"]


def export(target: str) -> None:
    src = OUT / f"s7_{target}_graded.jsonl"
    if not src.exists():
        print(f"{target}: no graded file, skipped")
        return
    dst = OUT / f"s7_{target}_samples.csv.gz"
    n = 0
    with gzip.open(dst, "wt", newline="", compresslevel=9) as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for line in open(src):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("extracted") is not None:
                # the graded answer span, not free text, but cap it defensively
                r["extracted"] = str(r["extracted"])[:120]
            w.writerow(r)
            n += 1
    print(f"{target}: {n} rows -> {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    for t in ("pool", "test"):
        export(t)
