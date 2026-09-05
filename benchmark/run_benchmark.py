"""Run every benchmark item against every tier at temperature 0.

Results append to raw_results/responses.jsonl one line at a time, so an
interrupted run resumes without re-billing completed calls.

Usage:
    python3 benchmark/run_benchmark.py --pilot 3     # cost estimation sample
    python3 benchmark/run_benchmark.py               # full 364 x 3
    python3 benchmark/run_benchmark.py --repeat 20   # determinism spot-check
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import MAX_TOKENS, MODELS, chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"
ITEMS_FILE = RAW / "benchmark_items.json"
RESPONSES_FILE = RAW / "responses.jsonl"
REPEAT_FILE = RAW / "responses_repeat.jsonl"

CONCURRENCY = {"together": 8, "openai": 8}


def load_items() -> list[dict]:
    with open(ITEMS_FILE) as f:
        return json.load(f)["items"]


def load_done(path: Path) -> set[tuple[int, str]]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None and r.get("answer"):
                    done.add((r["tier"], r["item_id"]))
    return done


async def worker(name, queue, client, out_file, lock, state):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            return
        tier, item = task
        t0 = time.time()
        try:
            res = await chat(
                client,
                tier,
                item["prompt"],
                MAX_TOKENS[item["benchmark"]],
                temperature=0.0,
            )
        except SystemExit as e:
            state["fatal"] = str(e)
            queue.task_done()
            return
        row = {
            "tier": tier,
            "model": MODELS[tier]["model"],
            "item_id": item["item_id"],
            "benchmark": item["benchmark"],
            "subject": item.get("subject"),
            "prompt": item["prompt"],
            "reference": item.get("reference"),
            "latency_s": round(time.time() - t0, 2),
            "repeat_index": state.get("repeat_index", 0),
            **res,
        }
        async with lock:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            state["n"] += 1
            if res["error"]:
                state["errors"] += 1
            state["usd"] += res["usd"] or 0.0
            if state["n"] % 25 == 0:
                print(
                    f"  {state['n']}/{state['total']} done, "
                    f"{state['errors']} errors, ${state['usd']:.3f}",
                    flush=True,
                )
        queue.task_done()


async def run(tasks: list[tuple[int, dict]], path: Path, repeat_index: int = 0) -> dict:
    state = {"n": 0, "total": len(tasks), "errors": 0, "usd": 0.0, "fatal": None,
             "repeat_index": repeat_index}
    lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)

    n_workers = CONCURRENCY["together"] + CONCURRENCY["openai"]
    for _ in range(n_workers):
        queue.put_nowait(None)

    limits = httpx.Limits(max_connections=n_workers + 4, max_keepalive_connections=n_workers)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(path, "a") as out_file:
            workers = [
                asyncio.create_task(worker(i, queue, client, out_file, lock, state))
                for i in range(n_workers)
            ]
            await asyncio.gather(*workers)
    if state["fatal"]:
        print(f"FATAL: {state['fatal']}", file=sys.stderr)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                    help="N items per benchmark per tier, for cost estimation")
    ap.add_argument("--repeat", type=int, default=0,
                    help="re-run the first N items of each benchmark a 2nd time")
    ap.add_argument("--tier", type=int, action="append",
                    help="restrict to specific tiers")
    args = ap.parse_args()

    items = load_items()
    tiers = sorted(set(args.tier)) if args.tier else [1, 2, 3]

    if args.pilot:
        selected, seen = [], {}
        for it in items:
            b = it["benchmark"]
            if seen.get(b, 0) < args.pilot:
                seen[b] = seen.get(b, 0) + 1
                selected.append(it)
        items = selected
        path = RAW / "responses_pilot.jsonl"
        done = load_done(path)
    elif args.repeat:
        selected, seen = [], {}
        for it in items:
            b = it["benchmark"]
            if seen.get(b, 0) < args.repeat:
                seen[b] = seen.get(b, 0) + 1
                selected.append(it)
        items = selected
        path = REPEAT_FILE
        done = load_done(path)
    else:
        path = RESPONSES_FILE
        done = load_done(path)

    RAW.mkdir(parents=True, exist_ok=True)
    tasks = [(t, it) for it in items for t in tiers if (t, it["item_id"]) not in done]
    print(f"{len(items)} items x {len(tiers)} tiers; {len(done)} already done; "
          f"{len(tasks)} calls to make -> {path.name}")
    if not tasks:
        return

    t0 = time.time()
    state = asyncio.run(run(tasks, path, repeat_index=1 if args.repeat else 0))
    print(f"\ncalls={state['n']} errors={state['errors']} "
          f"cost=${state['usd']:.4f} wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
