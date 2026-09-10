# WOAIS offline results

Derived from frozen Stage 1–2 / Stage 7 / Stage 9 artifacts. No API calls.
Frozen SHA256 checks passed before and after this run (`hash_check_*.json`).

## Cost and static baselines (n = 364, measured USD)

| Policy | Accuracy | Cost (USD) | Mean latency (s) | p95 latency (s) |
|---|---:|---:|---:|---:|
| EcoLogic keyword router | 86.8% | $0.2867 | 36.20 | 192.41 |
| Always Tier 1 | 87.6% | $0.3342 | 42.47 | 194.44 |
| Always Tier 2 | 92.3% | $0.0418 | 6.41 | 14.23 |
| Random tier | 90.7% | $0.3628 | 17.02 | 71.69 |
| Always-frontier (gpt-4o) | 91.5% | $0.6366 | 1.14 | 2.42 |
| Oracle (cheapest correct) | 96.4% | $0.0470 | 5.28 | 13.17 |

Always-Tier-2 dominates the router on accuracy, dollars, **and** wall-clock latency: 92.3% vs 86.8% at 1/6.87 of the cost and 5.6× lower mean wall-clock HTTP RTT (6.4s vs 36.2s). That RTT is not exclusive service time (it includes provider queueing).

The accuracy-optimal query-independent mixture at the router's realised budget is **always Tier 2** (mix weight 1.0, accuracy 92.3%). Extra spend does not buy static accuracy because Tier 3 is both costlier and less accurate than Tier 2 on this set.

## Naive vs exact cost

Pricing the keyword router as (tier mix) × (per-tier mean USD) understates its true per-item cost by 7.0% ($0.000733 vs $0.000788). The bias runs in the router's favour, matching the Stage 8/9 mechanism.

The regret identity reconciles on this table: R_naive = 0.000544569, correction = 0.000114014, R_true = 0.000658583 (residual 2.17e-19).

## Serverless-style queueing (model, not measurement)

Service times in the queueing model are stored `latency_s` HTTP round trips, not exclusive compute service. Arrival traces are synthetic Poisson processes over the frozen items. Cold-start extras are sensitivity knobs; TTFT was never recorded. Linear latency-vs-tokens R² is T1=0.979, T2=0.995, T3=0.920, so most wall-clock time is explained by completion length, not a residual cold-start. Simulated sojourn **double-counts** historical provider delay already inside `latency_s`.

At equal offered traffic (arrival rate = 0.8 × Always-Tier-2's 1-server capacity):

| Policy | Mean HTTP RTT (s) | Utilisation | p95 sojourn (s, SIMULATED) |
|---|---:|---:|---:|
| EcoLogic keyword router | 36.2 | 1.000 | 54171.6 |
| Always Tier 1 | 42.1 | 1.000 | 65178.1 |
| Always Tier 2 | 6.6 | 0.807 | 277.8 |
| Always-frontier (gpt-4o) | 1.2 | 0.141 | 2.7 |
| Random tier | 15.2 | 0.999 | 13409.8 |

Routing most queries to the long-reasoning Tier 1 replica saturates a SIMULATED serverless worker that Always-Tier-2 would keep at ~80% utilisation. This is a modelled capacity result, not a measured cloud deployment, and it replays already-recorded HTTP round-trips.

## Router overhead break-even vs cost-matched static

Maximum per-query router overhead (USD, latency, tokens) before a query-dependent assignment stops beating the cheapest static mix at the same quality. Analytic hull interpolation is checked by inverting the equal-cost envelope; percentile bootstrap CIs resample queries.

| Router | Axis | Raw savings | Overhead | Net | Break-even | % of break-even used | Status |
|---|---|---:|---:|---:|---:|---:|---|
| EcoLogic | usd | -0.000673066 | 0 | -0.000673066 | -0.000673066 | n/a | dominated |
| EcoLogic | latency_ms | -29783.5 | 0 | -29783.5 | -29783.5 | n/a | dominated |
| EcoLogic | tokens | -2470.07 | 0 | -2470.07 | -2470.07 | n/a | dominated |
| Always-T2 | usd | 0 | 0 | 0 | 0 | 0.0 | neutral |
| Always-T2 | latency_ms | 0 | 0 | 0 | 0 | 0.0 | neutral |
| Always-T2 | tokens | 0 | 0 | 0 | 0 | 0.0 | neutral |
| Oracle | usd | +inf | 0 | +inf | +inf | 0.0 | beneficial |
| Oracle | latency_ms | 1128.98 | 0 | 1128.98 | 1128.98 | 0.0 | beneficial |
| Oracle | tokens | 95.9588 | 0 | 95.9588 | 95.9588 | 0.0 | beneficial |

EcoLogic vs quality-matched static is **dominated** on USD (raw savings -0.000673066 $/query; keyword overhead 0; net -0.000673066). Latency overhead is unmeasured in frozen logs and treated as 0 (break-even -29783.5 ms; status dominated). Token overhead in config is 0 (break-even -2470.07 tokens; status dominated).

Always-T2 is the hull vertex, so USD status is **neutral** with break-even 0. The unconstrained oracle sits above the static hull (USD status **beneficial**, break-even +inf).

USD status counts: beneficial=1, neutral=1, dominated=4.

## Robustness of the EcoLogic-vs-static conclusion

Primary claim: EcoLogic is **dominated** vs the quality-matched static hull on realized USD. Hull-status labels changed in 2 non-bootstrap scenarios (0 reversed the claim to beneficial; 3 lost p<0.05 vs Always-T2 while remaining dominated). Bootstrap resample flip rate (status ≠ baseline): 0.0%. Significance is a paired sign-flip p-value vs Always-T2 quality, not CI overlap. The domination claim **does not survive** as a beneficial reversal under the hull perturbations. Tidy table: `routing/robustness.csv`.

## RouteLLM (committed Stage 9 JSON, not re-run)

n = 1307. Mean edge vs cost-matched mixture = +0.57 pp; 8/9 interior points positive; 0 interior McNemar p_raw<0.05 (unadjusted across the dependent sweep). The sweep-wide sign count is descriptive, not a Type-I-controlled test (p = 0.0391; interior points share items).

Stage 7 test-set latency compact table available: True. Stage 1–2 mix: {'gsm8k': 100, 'humaneval': 164, 'mmlu': 100}.
