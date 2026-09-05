"""
Build the EcoLogic publication-grade benchmark item set.

HumanEval: all 164 problems (graded by executing official tests).
MMLU:      100 questions, 10 subjects x 10 items, seeded.
GSM8K:     100 problems from the test split, seeded.

Datasets are cached under CACHE_DIR and are not committed. The sampled item
set (benchmark_items.json) and the selection manifest are committed so the
run is reproducible.
"""

import gzip
import json
import os
import random
import urllib.request
from pathlib import Path

SEED = 20260905
N_MMLU_SUBJECTS = 10
N_MMLU_PER_SUBJECT = 10
N_GSM8K = 100

CACHE_DIR = Path(os.environ.get("BENCH_CACHE", "/tmp/bench_cache"))
OUT_DIR = Path(__file__).resolve().parent.parent / "raw_results"

SOURCES = {
    "HumanEval.jsonl.gz": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "mmlu_test.parquet": "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
}

LETTERS = ["A", "B", "C", "D"]


def fetch(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(SOURCES[name], path)
    return path


def load_humaneval() -> list[dict]:
    path = fetch("HumanEval.jsonl.gz")
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = []
    for r in rows:
        items.append({
            "item_id": r["task_id"],
            "benchmark": "humaneval",
            "prompt": (
                "Complete the following Python function. Return the complete "
                "function implementation inside a single ```python code block. "
                "Do not include tests or explanation.\n\n"
                f"```python\n{r['prompt']}```"
            ),
            "grading": "execute",
            "entry_point": r["entry_point"],
            "he_prompt": r["prompt"],
            "he_test": r["test"],
            "reference": None,
        })
    return items


def load_mmlu(rng: random.Random) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    path = fetch("mmlu_test.parquet")
    rows = pq.read_table(path).to_pylist()
    by_subject: dict[str, list[dict]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)

    subjects = sorted(by_subject)
    eligible = [s for s in subjects if len(by_subject[s]) >= N_MMLU_PER_SUBJECT]
    chosen_subjects = sorted(rng.sample(eligible, N_MMLU_SUBJECTS))

    items = []
    manifest = {}
    for subject in chosen_subjects:
        pool = by_subject[subject]
        idxs = sorted(rng.sample(range(len(pool)), N_MMLU_PER_SUBJECT))
        manifest[subject] = idxs
        for i in idxs:
            r = pool[i]
            choices = list(r["choices"])
            if len(choices) != 4:
                continue
            lettered = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
            items.append({
                "item_id": f"mmlu/{subject}/{i}",
                "benchmark": "mmlu",
                "subject": subject,
                "prompt": (
                    "Answer this multiple-choice question. Respond with only the "
                    "letter of the correct option (A, B, C, or D) on the last line "
                    'in the form "Answer: X".\n\n'
                    f"{r['question']}\n{lettered}"
                ),
                "grading": "mcq",
                "reference": LETTERS[int(r["answer"])],
                "choices": choices,
            })
    return items, {"subjects": chosen_subjects, "indices": manifest}


def load_gsm8k(rng: random.Random) -> tuple[list[dict], dict]:
    path = fetch("gsm8k_test.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    idxs = sorted(rng.sample(range(len(rows)), N_GSM8K))
    items = []
    for i in idxs:
        r = rows[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "item_id": f"gsm8k/{i}",
            "benchmark": "gsm8k",
            "prompt": (
                "Solve this math word problem. Show your work, then give the final "
                'numeric answer on the last line in the form "Answer: N".\n\n'
                f"{r['question']}"
            ),
            "grading": "numeric",
            "reference": gold,
        })
    return items, {"split": "test", "indices": idxs, "pool_size": len(rows)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    humaneval = load_humaneval()
    mmlu, mmlu_manifest = load_mmlu(rng)
    gsm8k, gsm8k_manifest = load_gsm8k(rng)

    items = humaneval + mmlu + gsm8k
    payload = {
        "seed": SEED,
        "counts": {
            "humaneval": len(humaneval),
            "mmlu": len(mmlu),
            "gsm8k": len(gsm8k),
            "total": len(items),
        },
        "selection_manifest": {"mmlu": mmlu_manifest, "gsm8k": gsm8k_manifest},
        "sources": SOURCES,
        "items": items,
    }
    out = OUT_DIR / "benchmark_items.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out} with {len(items)} items: {payload['counts']}")
    print("mmlu subjects:", ", ".join(mmlu_manifest["subjects"]))


if __name__ == "__main__":
    main()
