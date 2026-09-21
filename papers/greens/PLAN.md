# GREENS 2027 — Paper B research plan

Venue: 11th International Workshop on Green and Sustainable Software (GREENS’27),
co-located with ICSE 2027 (Dublin). Emerging research paper, ≤8 pages,
IEEE IEEEtran 10pt conference, single-anonymous.

This document is PAPER B under `papers/SPLIT_CONTRACT.txt`. It is an energy /
green-SE paper. It is not an audit-protocol paper (Paper A) and not a
latency/Pareto paper (Paper C).

## Research question

Do software systems that advertise energy-saving LLM routing actually reduce
modelled energy versus the cheapest competent *static* policy, and is that
conclusion robust when the (unmetered) energy-rate model is varied?

Working title:
**Green by Default? A Deployed Energy-Saving LLM Router Uses More Energy than Ignoring the Query**

## Green-SE claims (what we will argue)

1. **Wrong baseline as a metric-gaming failure.** Comparing a router to
   always-frontier (ChatGPT-default) is nearly uninformative: any policy that
   rarely uses the frontier will “save” energy. Green software metrics that
   admit this comparison can be gamed. The bar is the cheapest *competent*
   static policy on the same item set (here: Always-Tier-2).
2. **Architectural anti-pattern.** When the middle tier dominates both quality
   *and* modelled energy, query-conditioned escalation is a sustainability
   anti-pattern: extra energy, extra frontier calls, no quality gain.
   Guidance: discover a dominating static tier *before* adding a router.
3. **Reporting-axis failure.** On identical routing decisions, EcoLogic’s
   headline saving vs always-frontier is 88.6% in modelled joules and 55.0% in
   measured USD — a ~4× move in the residual cost (0.114× vs 0.450× of
   frontier). Choice of cost model is itself a green-measurement result.
4. **Sensitivity as first-class evidence, not a footnote.** Energy is a linear
   transform of real token counts under assumed rates 0.5 / 1.5 / 60 J per 1k
   tokens. The ±5× factorial (27 combinations) is an experiment about the
   *energy model*, not a disclaimer.

## Unique experiments (must appear as main results)

| Experiment | Source | Role |
|---|---|---|
| Policy energy×accuracy table | `raw_results/graded.jsonl` + `routing.json`; cross-check `EVALUATION.md`, `results_report.md` | EcoLogic 689 J / 86.8% vs Always-T2 394 J / 92.3% vs Always-T1 677 J / 87.6% vs frontier 6066 J / 91.5% vs oracle 386 J / 96.4% |
| ±5× rate-sensitivity band | Reconstruct 3³ factorial from item-level tokens × varied rates (also in `raw_results/analysis.json`) | Savings vs frontier: −164.6% to +98.8%; 3/27 sign flips. **Also** EcoLogic-vs-Always-T2 energy ranking under the same 27 cells (this paper’s extra axis). |
| Per-benchmark energy mix | Item-level tokens × rates, grouped by HumanEval / MMLU / GSM8K | Shows where the router’s extra joules come from. |
| Joules-vs-dollars gap | Energy table vs `raw_results/tables_cost.md` / stored `usd` fields | 88.6% modelled-J vs 55.0% USD vs frontier. |

## What we will NOT claim

- That a joule was metered (wattmeter, GPU telemetry, PUE, idle). **Primary threat.**
- That EcoLogic “saves 88% energy vs GPT-4o” is a sustainability success. That
  comparison is the *wrong* one; the paper’s point is that it is uninformative.
- That routing cannot work in general, or that RouteLLM fails (one sentence
  max; no GSM8K 9-point table).
- Wall-clock latency, serverless DES, audit-protocol identity, long
  pre-registration / AUC-ceiling section.
- That the learned held-out router beats Always-T2 (it does not; quality 0.8889
  equals Always-T2 per contract — mention at most in limitations).
- Paid Cloud Run / Lambda measurement.

## Threats (hostile-reviewer version)

1. **Unmetered energy (PRIMARY).** Token-linear model; rates unsourced in
   `backend/main.py`. Sensitivity bounds robustness to the *constants*, not
   validity of the functional form (batching, utilisation, idle, hardware).
2. **Substitute models.** Gemma 3N E4B / Apriel 1.6 15B retired; Qwen3.5-9B and
   gpt-oss-20b are larger *reasoning* models. Token inflation (11.5× vs
   frontier) is likely substitution, not intended stack.
3. **In-sample keyword router** on n=364; classifier evaluated on the same set
   it was not trained on but *was* designed against (keyword rules).
4. **n=364**, academic single-turn benchmarks, not production traffic.
5. **Single generation** per item; token counts vary at temperature 0.

## CFP mapping (GREENS 2027)

- Metrics and measures for sustainability-aware SE
- Architectural implications / anti-patterns for green software
- Green AI; sustainability of generative-AI-enabled applications
- Energy-efficient architecture and design choices

## Process

1. This plan.
2. `analysis.py` reconstructs tables/figures into `artifacts/` from committed
   files only. No API calls. Do not write into `raw_results/`, `router_v2/`,
   `stage7_10/`.
3. `main.tex` + `refs.bib` + `build.sh` + `NUMBERS.md`.
