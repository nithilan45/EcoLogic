"""Run EcoLogic's real production classifier over the benchmark item set.

Imports classify_prompt_local_nlp straight out of backend/main.py so the
routing decisions are the deployed system's, not a reimplementation.

Primary routing input is item["raw_query"] (what a user would type). The
wrapped prompt actually sent to the model is classified too, as a
sensitivity check on prompt-format contamination of the router.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"
sys.path.insert(0, str(ROOT / "backend"))

from main import classify_prompt_local_nlp  # noqa: E402


def classify(text: str) -> dict:
    r = classify_prompt_local_nlp(text)
    return {
        "tier": int(r.recommended_tier),
        "difficulty": r.difficulty,
        "risk": r.risk,
        "reason": r.reason,
    }


def main():
    with open(RAW / "benchmark_items.json") as f:
        items = json.load(f)["items"]

    out = {}
    for it in items:
        out[it["item_id"]] = {
            "benchmark": it["benchmark"],
            "raw": classify(it["raw_query"]),
            "wrapped": classify(it["prompt"]),
        }

    path = RAW / "routing.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    import collections
    for key in ("raw", "wrapped"):
        dist = collections.Counter((v["benchmark"], v[key]["tier"]) for v in out.values())
        print(f"--- {key} query routing")
        for b in ("humaneval", "mmlu", "gsm8k"):
            row = {t: dist.get((b, t), 0) for t in (1, 2, 3)}
            print(f"  {b:<10} tier1={row[1]:>3} tier2={row[2]:>3} tier3={row[3]:>3}")
        tot = collections.Counter(v[key]["tier"] for v in out.values())
        print(f"  ALL        tier1={tot.get(1,0):>3} tier2={tot.get(2,0):>3} tier3={tot.get(3,0):>3}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
