# WOAIS experiments

Offline, publication-grade follow-ups to the EcoLogic routing audit: **exact
cost accounting**, **static baselines**, **empirical latency**, and
**serverless-style queueing models**. All numbers are derived from already
committed generations. This tree never calls a model API and never writes
outside `woais_experiments/`.

Existing Stage 1–10 outputs are treated as **immutable**. Their SHA256 digests
live in `EXISTING_RESULTS_SHA256.txt`. Read `REPO_AUDIT.md` before changing
anything here.

## Layout

| Directory | Role |
|---|---|
| `accounting/` | Token, USD, and energy accounting; naive vs true policy cost |
| `routing/` | Static, oracle, random, keyword, and cost-matched mixture policies |
| `latency/` | Wall-clock `latency_s` summaries and serverless queueing models |
| `workloads/` | Frozen item-set loaders and synthetic arrival traces |
| `external/` | RouteLLM tables from committed Stage 9 JSON (no clone, no BERT) |
| `statistics/` | Wilson / McNemar wrappers and the Stage 8 regret identity |
| `figures/` | Matplotlib writers; PNGs go to `results/figures/` |
| `tests/` | Hash, immutability, identity, and published-number tests |
| `configs/` | Seeds, rates, serverless sensitivity knobs, published anchors |
| `results/` | **New** artifacts only |

## Run

Use **Python 3.10+** with numpy / scipy / matplotlib (the eval stack). The
system `python3` on some machines is 3.7; prefer `python3.13` if that is what
has the scientific packages:

```bash
python3.13 -m unittest discover -s woais_experiments/tests -t . -v
python3.13 -m woais_experiments.run_offline
```

`run_offline` verifies frozen hashes, writes JSON/CSV/PNG under `results/`,
then verifies hashes again.
