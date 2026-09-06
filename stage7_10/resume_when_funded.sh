#!/usr/bin/env bash
# Resume Stage 7 generation, looping until no pending calls remain.
#
# Together rate-limits Tier 1 (Qwen3.5-9B) dynamically and the in-request retry
# budget (5 attempts, ~60s of backoff) is exhausted under sustained pressure, so
# a single pass abandons some calls with HTTP 429. Abandoned calls cost nothing
# and are retried here: each pass re-enqueues only what is still missing, keyed
# on (tier, item_id, sample_idx), so the loop converges without repeating any
# completed call.
set -u
cd /workspace

probe() {
  python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error
req = urllib.request.Request(
    "https://api.together.xyz/v1/chat/completions",
    data=json.dumps({"model": "openai/gpt-oss-20b",
                     "messages": [{"role": "user", "content": "ok"}],
                     "max_tokens": 4, "temperature": 0}).encode(),
    headers={"Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY','')}",
             "Content-Type": "application/json",
             "User-Agent": "Mozilla/5.0 EcoLogicBenchmark/1.0"})
try:
    with urllib.request.urlopen(req, timeout=60):
        sys.exit(0)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}", flush=True)
    sys.exit(1 if e.code == 402 else 2)
except Exception as e:
    print(type(e).__name__, flush=True)
    sys.exit(2)
PY
}

pending() {  # $1 = pool|test ; prints remaining call count
  python3 - "$1" <<'PY'
import sys
sys.path.insert(0, "stage7_10"); sys.path.insert(0, "benchmark")
import s7_run
from pathlib import Path
target = sys.argv[1]
items = s7_run.load_items(target)
path = s7_run.OUT / s7_run.TARGETS[target]["out"]
done = s7_run.load_done(path)
print(sum(1 for it in items for t in (1, 2, 3) for s in range(s7_run.K)
          if (t, it["item_id"], s) not in done))
PY
}

DEADLINE=$(( $(date +%s) + 5400 ))
while :; do
  if probe; then echo "$(date -u +%H:%M:%S) credits live"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u +%H:%M:%S) STILL 402 after 90 min — stopping, not burning calls"; exit 1
  fi
  echo "$(date -u +%H:%M:%S) not funded yet, retry in 60s"; sleep 60
done

for target in pool test; do
  for pass in $(seq 1 40); do
    left=$(pending "$target")
    echo "=== $target pass $pass: $left calls pending ($(date -u +%H:%M:%S)) ==="
    if [ "$left" -eq 0 ]; then echo "$target complete"; break; fi
    # Cap each pass at 25 min so every pass rebuilds the connection pool from
    # scratch; observed throughput starts near 200 calls/min on a fresh client
    # and decays as sockets go stale, so recycling is much faster than waiting.
    timeout -k 20 1500 python3 stage7_10/s7_run.py --target "$target"
    sleep 20
  done
done

echo "=== generation finished at $(date -u +%H:%M:%S) ==="
for target in pool test; do
  echo "$target still pending: $(pending "$target")"
done
