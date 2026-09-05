"""Provider-side nondeterminism spot-check at temperature 0.

Compares raw_results/responses_repeat.jsonl against the matching rows of
raw_results/responses.jsonl: byte-identical answer text, identical completion
token count, and identical grade.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade import grade_all  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"


def index(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["tier"], r["item_id"])] = r
    return out


def main():
    graded_repeat = RAW / "graded_repeat.jsonl"
    grade_all(RAW / "responses_repeat.jsonl", graded_repeat)

    first = index(RAW / "graded.jsonl")
    second = index(graded_repeat)

    rows = []
    for key, b in sorted(second.items()):
        a = first.get(key)
        if a is None:
            continue
        rows.append({
            "tier": key[0],
            "item_id": key[1],
            "benchmark": b["benchmark"],
            "text_identical": (a.get("answer") or "") == (b.get("answer") or ""),
            "tokens_identical": a.get("completion_tokens") == b.get("completion_tokens"),
            "grade_identical": bool(a.get("correct")) == bool(b.get("correct")),
            "grade_first": bool(a.get("correct")),
            "grade_second": bool(b.get("correct")),
            "ct_first": a.get("completion_tokens"),
            "ct_second": b.get("completion_tokens"),
        })

    summary = {"n_pairs": len(rows), "by_tier": {}}
    for t in (1, 2, 3):
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        summary["by_tier"][str(t)] = {
            "n": len(sub),
            "text_identical": sum(r["text_identical"] for r in sub),
            "tokens_identical": sum(r["tokens_identical"] for r in sub),
            "grade_identical": sum(r["grade_identical"] for r in sub),
            "grade_flips": [
                {"item_id": r["item_id"], "first": r["grade_first"], "second": r["grade_second"]}
                for r in sub if not r["grade_identical"]
            ],
        }
    summary["pairs"] = rows

    with open(RAW / "determinism.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(rows)} repeated (tier, item) pairs")
    print(f"{'tier':>4} {'n':>4} {'same text':>10} {'same tokens':>12} {'same grade':>11}")
    for t, s in summary["by_tier"].items():
        print(f"{t:>4} {s['n']:>4} {s['text_identical']:>10} "
              f"{s['tokens_identical']:>12} {s['grade_identical']:>11}")
    for t, s in summary["by_tier"].items():
        if s["grade_flips"]:
            print(f"tier {t} grade flips: {s['grade_flips']}")
    print(f"wrote {RAW / 'determinism.json'}")


if __name__ == "__main__":
    main()
