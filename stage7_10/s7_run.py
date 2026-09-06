"""Stage 7 generation - k=3 samples per item per tier.

Pool  : temperature 0.7 (samples the model's output distribution; temp-0 repeats
        are the least informative possible resamples)
Test  : temperature 0.0 (matches the main evaluation; the 3 samples also feed
        Stage 10(a)'s generation-variance decomposition)

Items are enqueued in pool order, so a prefix of completed items is a uniform
random subsample of the shuffled pool - the pre-registered throughput
contingency. Resumable on (tier, item_id, sample_idx).

Usage:
    python3 stage7_10/s7_run.py --target pool --pilot 6
    python3 stage7_10/s7_run.py --target pool
    python3 stage7_10/s7_run.py --target test
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

OUT = ROOT / "stage7_10"
K = 3
WORKERS = {"together": 80, "openai": 24}
TASK_TIMEOUT_S = 300

TARGETS = {
    "pool": {"files": ["s7_train_pool.json", "s7_calibration_pool.json"],
             "out": "s7_pool_responses.jsonl", "temperature": 0.7},
    "test": {"files": ["s7_test_set.json"],
             "out": "s7_test_responses.jsonl", "temperature": 0.0},
}


def cap_for(benchmark: str) -> int:
    return MAX_TOKENS.get(benchmark, MAX_TOKENS["humaneval"])


def load_items(target: str) -> list[dict]:
    """Load the target's items, interleaving splits proportionally.

    Enqueueing the TRAIN file before the CALIBRATION file means a truncated run
    yields TRAIN items only, which defeats the pre-registered contingency of
    falling back to the largest fully-generated prefix. Ordering each split by
    its fractional position and merging keeps any prefix representative of both
    splits.
    """
    per_split = []
    for name in TARGETS[target]["files"]:
        with open(OUT / name) as f:
            d = json.load(f)
        per_split.append([{**it, "split": d["split_name"]} for it in d["items"]])
    tagged = [(i / len(group), it)
              for group in per_split for i, it in enumerate(group)]
    tagged.sort(key=lambda p: p[0])
    return [it for _, it in tagged]


def load_done(path: Path) -> set:
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
                    done.add((r["tier"], r["item_id"], r["sample_idx"]))
    return done


async def worker(queue, client, out_file, lock, state, temperature):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            return
        tier, item, s = task
        t0 = time.time()
        try:
            # Hard ceiling per task. api.py retries up to 5 times at a 300s
            # request timeout, so a silently stalled connection can pin a worker
            # for ~25 minutes; with 80 workers that halts the whole run. Tier 1's
            # p90 latency is ~76s, so this ceiling never aborts a healthy call.
            # Abandoned tasks are re-enqueued by the next resume pass.
            res = await asyncio.wait_for(
                chat(client, tier, item["prompt"], cap_for(item["benchmark"]),
                     temperature=temperature),
                timeout=TASK_TIMEOUT_S)
        except SystemExit as e:
            state["fatal"] = str(e)
            queue.task_done()
            return
        except (asyncio.TimeoutError, TimeoutError):
            res = {"model": MODELS[tier]["model"], "content": "", "reasoning": "",
                   "answer": "", "http_status": None, "retries": 0,
                   "error": f"task_timeout after {TASK_TIMEOUT_S}s",
                   "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "usd": 0.0, "finish_reason": None}
        row = {
            "tier": tier, "model": MODELS[tier]["model"], "item_id": item["item_id"],
            "sample_idx": s, "benchmark": item["benchmark"], "subject": item.get("subject"),
            "split": item["split"], "temperature": temperature,
            "prompt": item["prompt"], "reference": item.get("reference"),
            "latency_s": round(time.time() - t0, 2), **res,
        }
        async with lock:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            state["n"] += 1
            if res["error"]:
                state["errors"] += 1
            state["usd"] += res["usd"] or 0.0
            if state["n"] % 250 == 0:
                el = time.time() - state["t0"]
                rate = state["n"] / el
                eta = (state["total"] - state["n"]) / rate / 60 if rate else 0
                print(f"  {state['n']}/{state['total']} done, {state['errors']} err, "
                      f"${state['usd']:.2f}, {rate * 60:.0f} calls/min, {eta:.0f} min left",
                      flush=True)
        queue.task_done()


async def run(tasks, path: Path, temperature: float) -> dict:
    state = {"n": 0, "total": len(tasks), "errors": 0, "usd": 0.0,
             "fatal": None, "t0": time.time()}
    lock = asyncio.Lock()
    queues = {p: asyncio.Queue() for p in WORKERS}
    for tier, item, s in tasks:
        queues[MODELS[tier]["provider"]].put_nowait((tier, item, s))
    for p, nw in WORKERS.items():
        for _ in range(nw):
            queues[p].put_nowait(None)

    total = sum(WORKERS.values())
    limits = httpx.Limits(max_connections=total + 16, max_keepalive_connections=total)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(path, "a") as out_file:
            ws = []
            for p, nw in WORKERS.items():
                for _ in range(nw):
                    ws.append(asyncio.create_task(
                        worker(queues[p], client, out_file, lock, state, temperature)))
            await asyncio.gather(*ws)
    if state["fatal"]:
        print(f"FATAL: {state['fatal']}", file=sys.stderr)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--pilot", type=int, default=0,
                    help="N items per benchmark, for cost/throughput estimation")
    args = ap.parse_args()

    cfg = TARGETS[args.target]
    items = load_items(args.target)
    if args.pilot:
        sel, seen = [], {}
        for it in items:
            b = it["benchmark"]
            if seen.get(b, 0) < args.pilot:
                seen[b] = seen.get(b, 0) + 1
                sel.append(it)
        items = sel
        path = OUT / f"s7_{args.target}_pilot.jsonl"
    else:
        path = OUT / cfg["out"]

    done = load_done(path)
    # item-major so a prefix of items completes first
    tasks = [(t, it, s) for it in items for t in (1, 2, 3) for s in range(K)
             if (t, it["item_id"], s) not in done]
    print(f"{len(items)} items x 3 tiers x k={K}; {len(done)} already done; "
          f"{len(tasks)} calls -> {path.name} (temperature {cfg['temperature']})")
    if not tasks:
        return
    st = asyncio.run(run(tasks, path, cfg["temperature"]))
    print(f"\ncalls={st['n']} errors={st['errors']} cost=${st['usd']:.4f} "
          f"wall={time.time() - st['t0']:.0f}s "
          f"rate={st['n'] / (time.time() - st['t0']) * 60:.0f} calls/min")


if __name__ == "__main__":
    main()
