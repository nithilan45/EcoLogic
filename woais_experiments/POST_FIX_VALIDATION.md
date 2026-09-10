# Post-fix validation

Date: 2026-09-10.

This report records the two evaluation bugfixes and the validation that
followed. Frozen hashed trees (`raw_results/`, `router_v2/`, `stage7_10/`)
were not modified.

**This checkout is not a fully clean publication artifact.** Unit tests
pass. Historical hashes match. Secrets are absent. Reconstruction and
clean-clone smoke succeed. `validate-artifact` still exits 1 because
`anonymity_scan` is critical on `git:log` (author names and emails in
Git history). That finding is not a code defect in the path or latency
fixes; it is not closable without rewriting Git history, which was not
done.

---

## Fixes

### 1. Path anonymization / portability

`woais_experiments/paths.py` no longer uses a host-prefix allowlist
(`_HOME_ABS_PREFIXES`, `/Users/`, `/home/`, …).

Serialization uses `pathlib` resolution and `relative_to` relationship
checks (`relative_to_root`). Any path inside the repository root becomes
a POSIX repository-relative path, including checkouts under `/mnt/data`,
`/tmp`, `/Users`, `/home`, and Windows drive/UNC syntax.

Paths outside the repository serialize as `<redacted-absolute>`
(`public_relpath` / `to_jsonable`).

Related user-facing strings (overwrite warnings, `ResultExistsError`,
run-directory exists errors) also go through `public_relpath`.

Tests: `woais_experiments/tests/test_paths.py` covers macOS-home, Linux-home, `/mnt/data/project`,
`/tmp/project`, and Windows-style paths. Host-root literals are built at
runtime so the test source does not itself trip the anonymity regexes.

### 2. Cold-like latency residual tolerance

`flag_cold_like` in `woais_experiments/latency/serverless.py` treated
`sd == 0` as the only “no outliers” case. A perfect linear fit leaves
residual standard deviation at machine precision, so `mu + k*sd` flagged
noise.

The detector now treats residual dispersion as zero when
`np.isclose(sd, 0.0, …)` or all residuals are `np.allclose` to the mean,
with an absolute floor that covers HTTP-RTT scale and large-|y|
least-squares crumbs.

Tests: `woais_experiments/tests/test_serverless.py`
(`TestColdLikeResidualTolerance`) cover perfectly linear timings, linear
timings plus machine-epsilon perturbation, one genuine outlier, multiple
genuine outliers, constant latency, extremely small values, and very
large values.

---

## Tests

Command (complete `woais_experiments` discovery, including
`external_routing/tests`):

```
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
  -m unittest discover -s woais_experiments -p 'test*.py' -t . -q
```

| | |
|---|---|
| Tests run | 426 |
| Passed | 426 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Elapsed | 21.111 s |
| Result | **OK** |

There is no other unittest/pytest tree at the repository root.

The clean-clone smoke (below) re-runs discovery limited to
`woais_experiments/tests` (415 collected; 1 skipped because the clone
has no `.git`; 0 failures).

---

## Historical hashes

Command:

```
python3.13 -c "from woais_experiments.frozen import verify_frozen_hashes; \
import json; print(json.dumps(verify_frozen_hashes(), indent=2))"
```

Manifest: `woais_experiments/EXISTING_RESULTS_SHA256.txt`.

| | |
|---|---|
| Records checked | 67 |
| Mismatches | 0 |
| Missing | 0 |
| Result | **ok** |

No frozen experimental output files were rewritten.

---

## Artifact validator

Command (full clean clone; not `--skip-clone`):

```
python3.13 run_woais.py validate-artifact --run-id post_fix_validate --no-symlink
```

Exit code: **1**.

Compact result:

| Field | Value |
|---|---|
| `ok` | false |
| `critical` | `anonymity_scan` |
| Secret hits | 0 |
| Reconstruction | **ok** (`n_failed=0`) |
| External public smoke | **ok** (RouteLLM GSM8K table, `n_items=1307`) |
| Package check | **ok** (`n_critical=0`, `n_warnings=7`) |
| Clean-clone smoke | **ok** (`skipped=false`, `exercised=true`, `checks_ok=true`) |
| Frozen hashes (inside validator) | **ok**, 67 files, 0 mismatch, 0 missing |

Written under `woais_experiments/results/reproducibility/`.

### Secrets

`secret_scan.json`: **ok**, 0 hits, 717 files scanned.

### Anonymity

`anonymity_scan.json` on this Git checkout:

| | |
|---|---|
| `ok` | false |
| Severity | critical (config default) |
| Hits | 50 |
| Files requiring review | 1: `git:log` |

Working-tree source and result files did not require review. All 50 hits
are Git log author/email lines. A git-less copy (the clean-clone tree)
does not scan `git:log`; nested `--skip-clone` validation inside that
clone reported `anonymity_files: 0`.

Git history was not rewritten.

### Reconstruction

`reproduce_tables.reconstruct()` succeeded (`n_failed=0`). This confirms
the committed USD headlines / framework aggregates / RouteLLM summary
keys, not a re-derivation of every latency, energy, or stress table.

---

## Remaining warnings

Package-check warnings (`n_warnings=7`, none critical):

- `.env.example` flagged as `env_template`
- duplicate-byte outputs:
  - `hash_check_before.json` / `hash_check_after.json`
  - `external_router/accounting_flips.csv` / `ranking_flips.csv`
  - accounting framework sign-flip vs naive over/underestimate CSV/JSON pairs

These are duplicate-content notices, not hash mismatches of frozen trees.

---

## Completion checklist (this checkout)

| Requirement | Status |
|---|---|
| 0 failing tests | **met** (426/426) |
| 0 historical hash mismatches | **met** (67/67) |
| No secrets detected | **met** |
| No critical anonymity failures | **not met** (`git:log` only) |
| Clean-clone smoke test succeeds | **met** |
| Aggregate reconstruction succeeds | **met** |

`validate-artifact` overall `ok` is **false**. Do not treat this tree as
a clean double-blind artifact until Git identity is removed from
history (or the published bundle is git-less).
