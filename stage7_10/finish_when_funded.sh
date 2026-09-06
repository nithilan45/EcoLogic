#!/usr/bin/env bash
# Finish Stage 7 and Stage 10(a) as soon as OpenAI credit is available.
#
# Everything except the frozen test set's Tier 3 column is already done. This
# waits for `gpt-4o` to accept a minimal call, then issues only the 1,091
# missing calls (resume is keyed on (tier, item_id, sample_idx)) and runs the
# rest of the pipeline unattended.
#
# Nothing here selects or tunes anything: the router weights, variant, threshold
# and seed are fixed on disk, so this only scores the frozen test set once.
set -u
cd /workspace

probe() {
  python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error
req = urllib.request.Request(
    "https://api.openai.com/v1/chat/completions",
    data=json.dumps({"model": "gpt-4o",
                     "messages": [{"role": "user", "content": "ok"}],
                     "max_tokens": 4, "temperature": 0}).encode(),
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}",
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60):
        sys.exit(0)
except urllib.error.HTTPError as e:
    body = e.read().decode()[:200]
    print(f"HTTP {e.code} {body}", flush=True)
    sys.exit(1)
except Exception as e:
    print(type(e).__name__, flush=True)
    sys.exit(1)
PY
}

DEADLINE=$(( $(date +%s) + 86400 ))   # watch for 24h, then give up quietly
until probe; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u +%H:%M:%S) still unfunded after 24h — stopping"; exit 1
  fi
  echo "$(date -u +%H:%M:%S) gpt-4o not funded yet, retry in 120s"; sleep 120
done
echo "$(date -u +%H:%M:%S) OpenAI credit live — finishing Stage 7"

for pass in $(seq 1 12); do
  left=$(python3 - <<'PY'
import sys
sys.path.insert(0, "stage7_10"); sys.path.insert(0, "benchmark")
import s7_run
items = s7_run.load_items("test")
done = s7_run.load_done(s7_run.OUT / s7_run.TARGETS["test"]["out"])
print(sum(1 for it in items for t in (1, 2, 3) for s in range(s7_run.K)
          if (t, it["item_id"], s) not in done))
PY
)
  echo "=== test pass $pass: $left calls pending ==="
  [ "$left" -eq 0 ] && break
  timeout -k 20 1500 python3 stage7_10/s7_run.py --target test
  sleep 20
done

set -e
python3 stage7_10/s7_grade.py --target test
python3 stage7_10/s7_final.py
python3 stage7_10/s7_variance.py
python3 stage7_10/s7_export_samples.py
echo "=== Stage 7 + 10(a) complete at $(date -u +%H:%M:%S) ==="
