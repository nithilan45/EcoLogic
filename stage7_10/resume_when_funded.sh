#!/usr/bin/env bash
# Poll Together AI until a minimal call succeeds, then resume Stage 7 generation.
# Both runs resume from the existing jsonl keyed on (tier, item_id, sample_idx),
# so no completed call is repeated.
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

DEADLINE=$(( $(date +%s) + 5400 ))   # give the balance up to 90 min to propagate
while :; do
  if probe; then
    echo "$(date -u +%H:%M:%S) CREDITS LIVE — resuming"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u +%H:%M:%S) STILL 402 after 90 min — giving up, not burning calls"
    exit 1
  fi
  echo "$(date -u +%H:%M:%S) not funded yet, retrying in 60s"
  sleep 60
done

echo "=== resuming pool (temp 0.7, k=3) ==="
python3 stage7_10/s7_run.py --target pool
echo "=== resuming test set (temp 0, k=3) ==="
python3 stage7_10/s7_run.py --target test
echo "=== generation done ==="
