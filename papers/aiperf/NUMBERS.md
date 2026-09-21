# NUMBERS.md — every reported digit → source

Policy-level latency method: **per-query selected-tier `latency_s`** (not mix-weighted).
Mix-weighted EcoLogic mean is reported only as a bias diagnostic.

Honesty: `latency_s` is full HTTP round-trip. TTFT was not recorded.
`woais_experiments/results/latency/stage12_wallclock.json` field `"honesty"`.

Reconstruction: `papers/aiperf/analysis.py` → `papers/aiperf/artifacts/`
(0 mismatches vs committed wall-clock + USD + frontier CSV).

## Panel

| Digit | Value | Source |
|---|---|---|
| n | 364 | `raw_results/graded.jsonl` complete 3-tier matrix; `artifacts/summary.json` |
| HumanEval / MMLU / GSM8K | 164 / 100 / 100 | split contract; `by_tier_benchmark` counts in `stage12_wallclock.json` |
| EcoLogic mix T1/T2/T3 | 274 / 87 / 3 | `routing.json` `raw`; `artifacts/mix_decomposition.json`; `raw_results/tables_cost.md` |
| Models | Qwen3.5-9B / gpt-oss-20b / gpt-4o | `frontier_aggregates.json` cheap/strong names; `graded.jsonl` `model` |

## Per-tier latency (n=364 each)

From `stage12_wallclock.json` `by_tier_benchmark.by_tier` and `artifacts/tier_latency.csv` (match).

| Tier | mean | p50 | p90 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|
| T1 | 42.47277… → **42.47 s** | **18.01** | **166.58** | 194.44 | **221.03** | 2.58 | 268.88 |
| T2 | 6.41321… → **6.41 s** | **4.19** | **9.96** | 14.23 | **34.46** | 0.53 | 195.96 |
| T3 | 1.14168… → **1.14 s** | **1.00** | **2.12** | 2.42 | **2.72** | 0.32 | 3.34 |

T1 mean / p50 = 42.47277/18.01 = **2.36×**.

T1 p90 / T2 p90 = 166.58/9.96 = **16.72×** (paper: $16.7\times$).

## Seconds per output token + tokens vs latency

`stage12_wallclock.json` `seconds_per_output_token`; `accounting/stage12.json` `corr_tokens_latency` + `token_dispersion`.

| Tier | s/tok mean | s/tok p50 | completion mean | Pearson r |
|---|---|---|---|---|
| T1 | 0.011936 → **11.94 ms** | 0.01172 | **3572.04** | **0.9896 → 0.990** |
| T2 | 0.012739 → **12.74 ms** | 0.01243 | **524.22** | **0.9977 → 0.998** |
| T3 | 0.044250 → **44.25 ms** | **0.008571 → 8.57 ms** | **140.60** | **0.9591 → 0.959** |

T1/T2 completion ratio 3572.04/524.22 = **6.81×**.

T3 ms/tok mean vs p50: HTTP-overhead note in Table I footnote.

## Policy latency (per-query)

`stage12_wallclock.json` `policies.*` and `artifacts/policy_cost_latency.csv`.

| Policy | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| EcoLogic | **36.1967 → 36.20 s** | **11.89** | **108.58** | 192.41 | **221.03** |
| Always-T1 | 42.47 | 18.01 | 166.58 | 194.44 | 221.03 |
| Always-T2 | **6.4132 → 6.41 s** | 4.19 | **9.96** | 14.23 | 34.46 |
| Always-T3 (`frontier`) | **1.1417 → 1.14 s** | 1.00 | 2.12 | 2.42 | 2.72 |
| Random | **17.0225 → 17.02 s** | 3.34 | 43.31 | 71.69 | 201.25 |
| Oracle USD | **5.2842 → 5.28 s** | 4.07 | 9.28 | 13.17 | 21.87 |

Ratios EcoLogic / Always-T2: mean 36.1967/6.4132 = **5.64×**; p50 11.89/4.19 = **2.84×**; p90 108.58/9.96 = **10.90×**.

## Mix-weighted vs realized (EcoLogic)

`artifacts/mix_decomposition.json`

| Quantity | Value |
|---|---|
| Mix-weighted mean | **33.5135 s** |
| Realized mean | **36.1967 s** |
| Mix − realized | **−2.683 s** |
| Relative bias | **−7.41%** |
| T1 share of latency mass | **0.9619 → 96.2%** |
| T1 share of queries | 274/364 = **75.27%** |
| Slower than Always-T2 | **261/364 = 71.7%** |
| Mean paired Δ vs T2 | **+29.783 s** |
| Median paired Δ vs T2 | **+6.265 s** |
| Routed-T1 mean (n=274) | **46.254 s** |
| Unconditional T1 mean | **42.473 s** |

`frontier_per_query.csv`: 364 rows; mix 274/87/3; mean selected ms/1000 matches realized mean.

`frontier_aggregates.json`: `router_overhead.n_measured` = **0**, `n_missing` = **364**.

## USD and accuracy

`raw_results/tables_cost.md` and `accounting/stage12.json` `axis_usd.policies` / `reproduction_vs_published`.

| Policy | Acc | k/n | Wilson CI | USD |
|---|---|---|---|---|
| EcoLogic | **86.8%** | 316/364 | [82.9%, 89.9%] | **$0.2867** (0.28674677) |
| Always-T1 | 87.6% | 319/364 | [83.9%, 90.6%] | $0.3342 |
| Always-T2 | **92.3%** | 336/364 | [89.1%, 94.6%] | **$0.0418** (0.0417509) |
| Always-T3 | 91.5% | 333/364 | [88.2%, 93.9%] | $0.6366 (0.636595) |
| Random | 90.7% | 330/364 | [87.2%, 93.2%] | $0.3628 |
| Oracle | 96.4% | 351/364 | [94.0%, 97.9%] | $0.0470 |

Cost ratio EcoLogic/T2 = 0.28674677/0.0417509 = **6.868**.

McNemar vs EcoLogic (`artifacts/summary.json`; matches `tables_cost.md`):
- Always-T2: b=5, c=25, **p=0.0003249 → 3.25e-4**
- Always-T1: b=0, c=3, p=0.25
- Always-T3: b=11, c=28, p=0.009475
- Random: b=5, c=19, p=0.006611

Random mix 125/107/132; oracle mix 9/343/12: `policy_cost_latency.csv`.

## Pareto

`artifacts/pareto.json`: deployable 2-D front `{always_t2, frontier}`; Always-T2 3-D-dominates EcoLogic = true.

## Wrapped surface (mentioned, not Table 1 shopping)

`stage12_wallclock.json` `policies.ecologic_wrapped`: mean 18.52 s, mix 137/53/174.
`axis_usd` wrapped: acc 0.896, USD 0.440.

## Tail mass, per-benchmark SLO, M/G/1 what-if

From `artifacts/tail_mass.csv`, `slo_by_benchmark.csv`, `mg1_whatif.csv`.

Top-10% latency mass: EcoLogic **54.4%** of selected-tier seconds in 37 queries; Always-T2 41.5%; Always-T3 22.1%.

Per-benchmark ≤10s: EcoLogic HE/MMLU/GSM **47.6 / 42 / 42%**; Always-T2 **86.6 / 96 / 90%**; Always-T3 100% all.

M/G/1 sojourn (hypothetical dedicated replica, RTT as exclusive S):
EcoLogic 10/50/90 q/h: **43.2 / 99.6 / 633 s**; unstable at 200.
Always-T2: **6.72 / 8.06 / 9.62 / 15.7 s**.
Honesty: not a prediction of the HTTP trace.

- Energy/joules (Paper B).
- RouteLLM GSM8K n=1307 (Paper A).
- Serverless DES sojourn times (`serverless_model.json`, honesty SIMULATED, double-counts HTTP RTT).
- Cloud Run/Lambda; `deployment_real_v2` local stub.
