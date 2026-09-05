"""Stage 1 - run all three tiers over the training/calibration pool.

Same generation config as the frozen-test-set run: temperature 0, 16,384-token
cap (reused from benchmark/api.py, not re-declared, so it cannot drift).

Writes only into router_v2/. Nothing in raw_results/ is read for generation or
written at all. Appends one JSON line per call so an interrupted run resumes
without re-billing completed calls.

Usage:
    python3 router_v2/run_pool.py --pilot 4
    python3 router_v2/run_pool.py
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from api import MAX_TOKENS, MODELS, chat  # noqa: E402

OUT = ROOT / "router_v2"
POOL_RESPONSES = OUT / "pool_responses.jsonl"
PILOT_RESPONSES = OUT / "pool_responses_pilot.jsonl"

# separate worker pools per provider so Together's dynamic rate limit is not
# hammered by OpenAI-bound slack capacity
WORKERS = {"together": 20, "openai": 12}


def load_pool() -> list[dict]:
    items = []
    for name in ("train_pool.json", "calibration_pool.json"):
        with open(OUT / name) as f:
            d = json.load(f)
        for it in d["items"]:
            items.append({**it, "split": d["split_name"]})
    return items


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


def cap_for(benchmark: str) -> int:
    # mbpp is a code benchmark; give it the same cap HumanEval got
    return MAX_TOKENS.get(benchmark, MAX_TOKENS["humaneval"])


async def worker(queue, client, out_file, lock, state):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            return
        tier, item = task
        t0 = time.time()
        try:
            res = await chat(client, tier, item["prompt"],
                             cap_for(item["benchmark"]), temperature=0.0)
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
            "split": item["split"],
            "prompt": item["prompt"],
            "reference": item.get("reference"),
            "latency_s": round(time.time() - t0, 2),
            **res,
        }
        async with lock:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            state["n"] += 1
            if res["error"]:
                state["errors"] += 1
            state["usd"] += res["usd"] or 0.0
            if state["n"] % 100 == 0:
                el = time.time() - state["t0"]
                rate = state["n"] / el
                eta = (state["total"] - state["n"]) / rate / 60 if rate else 0
                print(f"  {state['n']}/{state['total']} done, {state['errors']} errors, "
                      f"${state['usd']:.3f}, {eta:.0f} min left", flush=True)
        queue.task_done()


async def run(tasks, path: Path) -> dict:
    state = {"n": 0, "total": len(tasks), "errors": 0, "usd": 0.0,
             "fatal": None, "t0": time.time()}
    lock = asyncio.Lock()
    queues = {p: asyncio.Queue() for p in WORKERS}
    for tier, item in tasks:
        queues[MODELS[tier]["provider"]].put_nowait((tier, item))
    for p, n in WORKERS.items():
        for _ in range(n):
            queues[p].put_nowait(None)

    total_workers = sum(WORKERS.values())
    limits = httpx.Limits(max_connections=total_workers + 8,
                          max_keepalive_connections=total_workers)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(path, "a") as out_file:
            ws = []
            for p, n in WORKERS.items():
                for _ in range(n):
                    ws.append(asyncio.create_task(worker(queues[p], client, out_file, lock, state)))
            await asyncio.gather(*ws)
    if state["fatal"]:
        print(f"FATAL: {state['fatal']}", file=sys.stderr)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                    help="N items per benchmark per tier, for cost estimation")
    args = ap.parse_args()

    items = load_pool()
    if args.pilot:
        sel, seen = [], {}
        for it in items:
            b = it["benchmark"]
            if seen.get(b, 0) < args.pilot:
                seen[b] = seen.get(b, 0) + 1
                sel.append(it)
        items, path = sel, PILOT_RESPONSES
    else:
        path = POOL_RESPONSES

    done = load_done(path)
    tasks = [(t, it) for it in items for t in (1, 2, 3) if (t, it["item_id"]) not in done]
    print(f"{len(items)} items x 3 tiers; {len(done)} already done; "
          f"{len(tasks)} calls to make -> {path.name}")
    if not tasks:
        return
    st = asyncio.run(run(tasks, path))
    print(f"\ncalls={st['n']} errors={st['errors']} cost=${st['usd']:.4f} "
          f"wall={time.time() - st['t0']:.0f}s")


if __name__ == "__main__":
    main()
