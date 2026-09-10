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
| `routing/` | Static, oracle, random, keyword, cost-matched, and robustness sweeps |
| `latency/` | Wall-clock `latency_s` summaries and serverless queueing models |
| `workloads/` | Frozen item-set loaders and synthetic arrival traces |
| `external/` | Generic routing-benchmark adapter + RouteLLM Stage 9 tables (no download, no BERT) |
| `statistics/` | Wilson / McNemar wrappers and the Stage 8 regret identity |
| `figures/` | Matplotlib writers; PNGs go to `results/figures/` |
| `tests/` | Hash, immutability, identity, and published-number tests |
| `configs/` | Seeds, rates, serverless sensitivity knobs, published anchors |
| `reproducibility/` | Clean-clone, table reconstruction, secret and anonymity scans |
| `results/` | **New** artifacts only |

## Run

Use **Python 3.10+** with numpy / scipy / matplotlib (the eval stack). The
system `python3` on some machines is 3.7; prefer `python3.13` if that is what
has the scientific packages:

```bash
python3.13 -m unittest discover -s woais_experiments/tests -t . -v
python3.13 -m woais_experiments.run_offline
python3.13 run_woais.py audit
python3.13 run_woais.py all
```

`run_offline` verifies frozen hashes, writes JSON/CSV/PNG under `results/`,
then verifies hashes again.

`run_woais.py` is the reproducible CLI: YAML config (`configs/run.yaml`),
deterministic seeds, timestamped `results/runs/<utc>_<suffix>/` directories,
a `results/runs/latest` symlink when the OS allows it, resume, JSONL logs,
git commit + config hash on every stage and artifact, and no paid API calls
unless `--allow-api` is passed. Existing run directories are never overwritten
silently (`--resume` or `--force` required). Prerequisites (Python version,
frozen hashes, required files) are checked before a stage starts.

```bash
python3.13 run_woais.py {audit|external|accounting|static|oracle|latency|workload|robustness|all|validate-artifact}
python3.13 run_woais.py audit                          # new timestamped dir
python3.13 run_woais.py all --resume                   # continue results/runs/latest
python3.13 run_woais.py accounting --force --run-id ID # replace artifacts in ID
python3.13 run_woais.py validate-artifact              # clean-clone + secret/anonymity scans
python3.13 run_woais.py validate-artifact --skip-clone # scans + table reconstruction only
```

`all` runs audit → accounting → static → oracle → latency → workload →
external → robustness. Latency draws `n_random_sims` from the YAML (default
2000); that stage is slow. Each run prints a concise summary and writes
`summary.json` plus `run.log.jsonl` / `artifacts.jsonl` in the run directory.

`validate-artifact` does **not** open a timestamped experiment run. It copies
the tree into a fresh venv (unless `--skip-clone`), reconstructs headline
tables from raw query-level files, and writes secret/anonymity/package
reports under `results/reproducibility/`. It never rewrites stored Stage 1–12
tables. The process exits non-zero when a critical check fails (including
identity hits configured in `reproducibility/anonymity_config.yaml`).
