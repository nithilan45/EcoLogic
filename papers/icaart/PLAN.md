# ICAART 2027 — Paper A plan

**Working title:** Does Looking at the Query Help? An Audit Protocol for LLM Routers

**Venue:** ICAART 2027 (CORE B), Agents & AI. SciTePress, double-blind regular paper.

**Community:** AI evaluation / LLM systems as decision procedures. Not green SE, not middleware, not latency.

---

## Research question

Does query-conditioned model selection beat a **cost-matched query-independent** policy, and can a protocol distinguish routers that earn their complexity from those that do not?

## Claims we will actually support

1. **Wrong baseline.** Comparing a router only to always-frontier is nearly uninformative. The bar a router must clear is a query-independent policy at the same expected cost (static single-tier, or a mixture whose expected cost matches the router).
2. **EcoLogic fails the audit (in-sample keyword).** On the frozen Stage 1–2 panel (n=364, measured USD), Always-Tier-2 dominates the production keyword router on both accuracy and dollars (86.8% / $0.2867 vs 92.3% / $0.0418; McNemar exact p=0.0003249). Caption must state **in-sample**.
3. **RouteLLM barely passes, below benchmark resolution.** On RouteLLM’s own GSM8K (n=1307), mean edge vs cost-matched mixture is +0.57 pp; 0/9 interior points are significant at p<0.05; sign test p=0.039 (descriptive; points are dependent). Call-fraction matching overstates the edge by 0.09–0.30 pp.
4. **Naive mix×mean cost is biased for routers.** Exact identity \(R_{\mathrm{true}}=R_{\mathrm{naive}}+\sum_{i\neq j}\mathrm{Cov}(1\{C_{ij}}, D_{ij})\). Sign of EcoLogic oracle-regret flips after correction; RouteLLM naive accounting overstates savings by up to 6.3%. The identity is exact for query-independent assignment.
5. **Held-out learned replacement is a null.** Pre-registered `heldout_v1`: selected logistic quality 0.8889, identical to always_cheap / cost_matched_static. Not a win.
6. **Ceiling is not “too weak a model class.”** Same MiniLM labels/splits: logistic CALIBRATION AUC 0.6816 beats GB/RF/kNN.
7. **Temperature 0 is not deterministic.** Tier 1 graded verdict flips on 19.8% of items (k=3); 47% of Always-T1 Wilson-interval width is regeneration noise.
8. **The protocol separates systems.** EcoLogic fails; RouteLLM directionally passes at an effect size the field’s n=1307 GSM8K panel cannot resolve pointwise.

## Unique experiments (this paper only)

Per `papers/SPLIT_CONTRACT.txt`:

- RouteLLM GSM8K cost-matched vs call-fraction matching (main **external** study).
- Pre-registered learned router + ceiling AUC (logistic vs GB/RF/kNN).
- Temp-0 regeneration noise / variance decomposition.
- Naive-vs-true cost identity as **evaluation bias**, illustrated on RouteLLM (not a green-SE story).

## What we will NOT claim

- Energy was measured; joules as a headline metric; ±5× energy-rate sweep as a main experiment.
- Wall-clock latency, p90, serverless DES, Cloud Run/Lambda (those files are stubs).
- “Routing cannot work” / all learned routers fail.
- Held-out logistic **beats** Always-T2 (it equals it).
- Stage 1–2 keyword evaluation is out-of-sample.
- RouteLLM’s quality-vs-call-fraction curves are wrong (they are not; only cost translation and the missing cost-matched baseline are).
- Generality beyond one zero-API-cost router family, substituted tiers, and academic auto-gradable items.

## Threats to validity

- **In-sample keyword (C1).** Keyword lists were not locked on a held-out split; Table of Stage 1–2 is an audit of the deployed heuristic on the panel used to inspect it.
- **Substitutes.** Original Together slugs retired; Qwen3.5-9B / gpt-oss-20b stand in for T1/T2. Routing *logic* is audited, not the original stack.
- **Hull is in-sample.** Cost-matched static on Stage 1–2 uses unconditional means on the same 364 items; on this price vector T2 dominates, so cost-matched static **is** Always-T2.
- **RouteLLM reconstruction.** Tokens from released response text + tokenizers; prices supplied by this study; BERT router only; GSM8K only (MMLU/MT-Bench lack response text).
- **Held-out pool reuse.** The 1200-item pool was used in earlier `router_v2/` work (method contamination from prior peeking not fully controlled).
- **Generation noise.** Single-run accuracies for long-output T1 contain regeneration variance; McNemar is paired on the same generations.
- **Sign test on RouteLLM sweep** is not an exact independent test (shared items).
- **No production traffic.** HumanEval/MMLU/GSM8K are not user logs.

## Section outline and budget

Target 30k–40k characters excluding whitespace (~10–12 SciTePress pages). Abstract ≤200 words.

| § | Content | ~chars ex. space |
|---|---------|------------------|
| 0 | Title, abstract, keywords | 1.5k |
| 1 | Introduction: evaluation practice, wrong baseline, wrong cost axis | 4.5k |
| 2 | Related work (RouteLLM, FrugalGPT, cascading, Mixtral routing — **evaluation**) | 4k |
| 3 | Audit protocol | 4k |
| 4 | Cost-accounting bias (proposition, sketch, numbers) | 4k |
| 5 | Study A: EcoLogic n=364 USD (**in-sample** caption) | 4k |
| 6 | Study B: RouteLLM GSM8K (unique main external) | 5k |
| 7 | Study C: held-out null + ceiling AUC + noise | 4.5k |
| 8 | Discussion: two systems, two outcomes; effect vs resolution | 3k |
| 9 | Limitations | 2.5k |
| 10 | Conclusion | 1.5k |
| — | References | — |

## Analysis reconstruction (offline)

`papers/icaart/analysis.py` rebuilds every table from committed JSON/CSV/JSONL. Writes only under `papers/icaart/artifacts/`. No API calls. Frozen trees `raw_results/`, `router_v2/`, `stage7_10/` are read-only.

## Honesty constraints (`FINAL_WOAIS_AUDIT.md`)

Null remains a null. Keyword in-sample caveat in the table caption. No paid deployment claim. Energy modelled if mentioned at all (prefer not to). Git history deanonymizes — paper is double-blind independently of the git artifact.
