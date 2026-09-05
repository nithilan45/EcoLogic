"""Objective auto-grading for all three benchmarks. No LLM judge anywhere.

humaneval : extract the function, run the official test suite in a resource
            limited subprocess, pass@1 = exit status 0
mmlu      : extract the answer letter, exact match against the gold letter
gsm8k     : extract the final number, numeric match against the gold answer

Writes raw_results/graded.jsonl (one row per response, grade attached).
"""

import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"

EXEC_TIMEOUT = 15
MEM_LIMIT_BYTES = 4 * 1024 * 1024 * 1024

SANDBOX_PREAMBLE = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (%d, %d))
resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))
resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
sys.setrecursionlimit(10000)
""" % (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES, EXEC_TIMEOUT, EXEC_TIMEOUT)


# ---------------------------------------------------------------- extraction

def extract_code(text: str, entry_point: str) -> str:
    """Pull the candidate implementation out of a chat response."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        # unterminated fence
        m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.DOTALL)
        if m:
            blocks = [m.group(1)]
    for b in blocks:
        if f"def {entry_point}" in b:
            return b
    if blocks:
        return blocks[0]
    if f"def {entry_point}" in text:
        return text
    return ""


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    m = list(re.finditer(r"answer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([ABCD])\)?\b",
                         text, re.IGNORECASE))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"^\s*\(?([ABCD])\)?\s*[\.\)]?\s*$", text, re.MULTILINE))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"\b([ABCD])\b", text))
    if m:
        return m[-1].group(1).upper()
    return None


NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _norm_num(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def extract_number(text: str):
    if not text:
        return None
    m = list(re.finditer(r"answer\s*(?:is)?\s*[:\-]?\s*\**\s*(" + NUM_RE + r")",
                         text, re.IGNORECASE))
    if m:
        return _norm_num(m[-1].group(1))
    m = list(re.finditer(r"####\s*(" + NUM_RE + r")", text))
    if m:
        return _norm_num(m[-1].group(1))
    m = list(re.finditer(NUM_RE, text))
    if m:
        return _norm_num(m[-1].group(0))
    return None


# ------------------------------------------------------------------ execution

def run_humaneval_one(args) -> dict:
    """Build and execute prompt-preamble + candidate + official tests."""
    code, he_prompt, he_test, entry_point = args
    if not code.strip():
        return {"correct": False, "exec_status": "no_code", "stderr": ""}

    preamble = he_prompt.split("def ")[0] if "def " in he_prompt else ""
    if f"def {entry_point}" in code:
        body = code
    else:
        body = he_prompt + code

    program = (
        SANDBOX_PREAMBLE
        + "\n" + preamble
        + "\n" + body
        + "\n\n" + he_test
        + f"\n\ncheck({entry_point})\nprint('__OK__')\n"
    )

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "cand.py"
        f.write_text(program)
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-S", str(f)],
                capture_output=True, text=True, timeout=EXEC_TIMEOUT + 5, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return {"correct": False, "exec_status": "timeout", "stderr": ""}
        ok = p.returncode == 0 and "__OK__" in p.stdout
        if ok:
            status = "pass"
        elif p.returncode != 0:
            status = "assert_or_error"
        else:
            status = "no_ok_marker"
        return {"correct": ok, "exec_status": status, "stderr": (p.stderr or "")[-600:]}


# --------------------------------------------------------------------- driver

def grade_all(in_path: Path, out_path: Path) -> list[dict]:
    with open(RAW / "benchmark_items.json") as f:
        items = {it["item_id"]: it for it in json.load(f)["items"]}

    rows = []
    seen = set()
    with open(in_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["tier"], r["item_id"], r.get("repeat_index", 0))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    he_jobs, he_idx = [], []
    for i, r in enumerate(rows):
        it = items[r["item_id"]]
        text = r.get("answer") or ""
        r["truncated"] = r.get("finish_reason") == "length"

        if it["benchmark"] == "humaneval":
            code = extract_code(text, it["entry_point"])
            r["extracted_code"] = code
            he_jobs.append((code, it["he_prompt"], it["he_test"], it["entry_point"]))
            he_idx.append(i)
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

    if he_jobs:
        print(f"executing {len(he_jobs)} HumanEval candidates ...", flush=True)
        with ProcessPoolExecutor(max_workers=8) as ex:
            for i, res in zip(he_idx, ex.map(run_humaneval_one, he_jobs)):
                rows[i]["correct"] = res["correct"]
                rows[i]["exec_status"] = res["exec_status"]
                rows[i]["exec_stderr"] = res["stderr"]
                rows[i]["gradable"] = res["exec_status"] != "no_code"

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return rows


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW / "responses.jsonl"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW / "graded.jsonl"
    rows = grade_all(src, dst)

    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[(r["tier"], r["benchmark"])]
        a[0] += int(bool(r.get("correct")))
        a[1] += 1
    for k in sorted(agg):
        c, n = agg[k]
        print(f"tier {k[0]} {k[1]:<10} {c}/{n} = {c / n:.1%}")


if __name__ == "__main__":
    main()
