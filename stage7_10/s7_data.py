"""Stage 7 shared loaders.

POOL (used for fitting and calibration) applies the pre-registered label rule:
per-tier correctness is the MAJORITY of the k=3 temperature-0.7 samples, and
per-tier energy uses the MEAN total tokens over those samples.

TEST (the one-shot frozen set) uses sample index 0 only for the headline
numbers, so the table is comparable to Stage 5's single temperature-0
generation per item. All three test samples are used only by Stage 10(a)'s
generation-variance decomposition.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from api import MODELS  # noqa: E402

OUT = ROOT / "stage7_10"
TIERS = [1, 2, 3]
K = 3
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}
BENCHMARKS = ["mbpp", "mmlu", "gsm8k"]


def load_pool_labels() -> dict:
    with open(OUT / "s7_pool_labels.json") as f:
        return json.load(f)


def pool_correct_and_tokens() -> tuple[dict, dict]:
    """Majority-vote correctness and mean-token counts, keyed (tier, item_id)."""
    lab = load_pool_labels()["labels"]
    correct, tokens = {}, {}
    for item_id, d in lab.items():
        for t in TIERS:
            correct[(t, item_id)] = bool(d["majority_correct_by_tier"][str(t)])
            tokens[(t, item_id)] = float(d["mean_tokens_by_tier"][str(t)])
    return correct, tokens


def load_split(name: str) -> list[dict]:
    """name in {train, calibration, test}; only items with complete k=3 labels."""
    fn = {"train": "s7_train_pool.json", "calibration": "s7_calibration_pool.json",
          "test": "s7_test_set.json"}[name]
    with open(OUT / fn) as f:
        items = json.load(f)["items"]
    if name == "test":
        return items
    labelled = set(load_pool_labels()["labels"])
    return [it for it in items if it["item_id"] in labelled]


def test_rows(sample_idx: int | None = None) -> list[dict]:
    rows = []
    with open(OUT / "s7_test_graded.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if sample_idx is None or r["sample_idx"] == sample_idx:
                    rows.append(r)
    return rows


def test_correct_and_tokens(sample_idx: int = 0) -> tuple[dict, dict, dict, list[str]]:
    """Per-item outcomes from one generation index, plus benchmark lookup."""
    items = load_split("test")
    bench_of = {it["item_id"]: it["benchmark"] for it in items}
    correct, tokens = {}, {}
    for r in test_rows(sample_idx):
        correct[(r["tier"], r["item_id"])] = bool(r.get("correct"))
        tokens[(r["tier"], r["item_id"])] = int(r.get("total_tokens") or 0)
    complete = sorted(i for i in bench_of
                      if all((t, i) in correct for t in TIERS))
    return correct, tokens, bench_of, complete
