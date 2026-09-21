# AIPerf 2027 — Paper C plan

**Working title:** Routing Is Not Free: Latency and Cost of Query-Conditioned LLM Assignment

**Venue:** AIPerf 2027 (Workshop on AI for Software Performance and Performance Engineering for AI Systems), co-located with ICSE 2027.

**Split-contract identity:** PAPER C. Primary axis is wall-clock HTTP `latency_s` plus measured USD. Energy/joules, RouteLLM GSM8K, green-SE language, AUC/temp-0 noise, and paid Cloud Run/Lambda claims are forbidden in abstract, title, and Table 1.

## Research question

On the cost–latency–accuracy frontier of LLM serving, does query-conditioned assignment improve the Pareto set over static single-tier policies?

## Performance claims (must be reconstructible; do not invent)

1. **Per-tier latency is not interchangeable.** On the frozen 364-item panel, T1 (Qwen3.5-9B) mean 42.47 s / p50 18.01 s / p90 166.58 s / p99 221.03 s; T2 (gpt-oss-20b) 6.41 / 4.19 / 9.96 / 34.46; T3 (gpt-4o) 1.14 / 1.00 / 2.12 / 2.72. T1’s mean is tail-dominated (mean ≫ median). `latency_s` is full HTTP round-trip, not TTFT.
2. **EcoLogic’s policy latency is a T1 mixture.** Raw keyword mix is 274/87/3 (T1/T2/T3). Policy end-to-end latency is the per-query selected-tier `latency_s`, not a mix×mean of tier averages. Reconstruct from the item matrix + `routing.json` `raw` assignments; cross-check `stage12_wallclock.json` and `frontier_per_query.csv`.
3. **Always-T2 Pareto-dominates EcoLogic on USD, latency, and accuracy.** EcoLogic $0.2867 / 86.8% vs Always-T2 $0.0418 / 92.3%, with EcoLogic mean/p50/p90 latency far above T2. Always-T3 is the low-latency, high-USD vertex. EcoLogic is interior: slower than both static vertices and more expensive than T2.
4. **Tokens, not slower tokens, explain T1 wall-clock.** Seconds per output token are similar for T1 and T2 (~12 ms); T1 emits ~3572 completion tokens vs T2 ~524. Pearson r(completion tokens, latency) ≈ 0.99 (T1) / 1.00 (T2) / 0.96 (T3).
5. **Router-span overhead was not measured** on this panel (`frontier_aggregates.json`: `n_measured = 0`). The paper’s “routing is not free” claim is the *assignment mix*, not a measured classifier hop.

## Unique experiments (main results)

| Experiment | Source | Role |
|---|---|---|
| Per-tier latency CDFs + percentiles | `graded.jsonl` via `ItemMatrix.latency_s`; committed `stage12_wallclock.json` | Figure 1 |
| Policy e2e latency (per-query selected tier) | matrix + `routing.json`; `frontier_per_query.csv` | Table 1 / §5 |
| Cost–latency scatter (bubble = accuracy) | USD from `evaluate_policies` + policy latency | Figure 2 |
| Tokens vs latency / s per output token | `corr_tokens_vs_latency`, `latency_per_output_token` | §5 |
| Mix×mean vs realized policy latency | same matrix | shows covariance, like USD identity |

**Serverless DES:** omit from main results. Optional one-paragraph threat: `serverless_model.json` is SIMULATED, replays HTTP RTT as service time then queues (double-count). `deployment_real_v2` is a local stub, not a cloud measurement.

## What this paper is *not*

- Not a green-SE / joule paper (Paper B).
- Not an audit-protocol / RouteLLM GSM8K / AUC-ceiling paper (Paper A).
- Not a claim that a paid Cloud Run or Lambda campaign was run.

## Threats to validity (must survive a hostile reviewer)

- HTTP RTT ≠ TTFT, ≠ GPU occupancy, ≠ isolated decode time; provider queueing is inside `latency_s`.
- No concurrent multi-tenant load; each call is sequential/offline.
- Keyword router is **in-sample** on n=364; Always-T2 is an in-sample static vertex.
- Substitute models (retired Together slugs).
- Router decision time unmeasured; EcoLogic latency understates a production router hop.
- No cloud deployment measurement.
- Oracle is hindsight (label-using); not deployable.
- T2 has a long tail too (max 196 s); means remain misleading without percentiles.

## Analysis contract

- Script: `papers/aiperf/analysis.py` writes only under `papers/aiperf/artifacts/`.
- No writes to `raw_results/`, `router_v2/`, `stage7_10/`.
- No API calls.
- Every digit in `main.tex` is listed in `NUMBERS.md` with a source path.
