# How to tell whether an energy-saving LLM router actually saves anything

**The goal of this research, in one sentence:** energy-saving query routers are
currently reported without the baselines or the cost accounting needed to know
whether they save anything — this work builds the audit protocol that makes such
a claim falsifiable, and applies it to a deployed router, which fails it.

That is the research contribution. EcoLogic is the subject of the audit, not the
point of it. If you only remember one thing: **the protocol is the contribution,
and the negative result is the evidence that the protocol bites.**

---

## Contents

1. [The question, and why it isn't already answered](#1-the-question)
2. [What this contributes](#2-what-this-contributes)
3. [What was found](#3-what-was-found)
4. [Where to read what](#4-where-to-read-what)
5. [How the numbers were produced](#5-how-the-numbers-were-produced)
6. [What this does NOT establish](#6-what-this-does-not-establish)
7. [Reproducing it](#7-reproducing-it)

---

## 1. The question

A tiered router sends easy queries to a small model and hard ones to a frontier
model, then reports the energy it saved versus sending everything to the frontier
model. That comparison is the standard one, and it is nearly uninformative,
because **it is not the bar the router has to clear.** Two much cheaper policies
also beat always-frontier on energy: "always use the small model" and "always use
the middle model." Neither looks at the query at all. A router only earns its
complexity if it beats *those*.

So the question this work asks is not "does the router save energy versus the
frontier" — it will, trivially — but:

> **Does looking at the query beat not looking at the query?**

This began as a narrower task. The audited system's own limitations section
conceded:

> "no formal quality evaluation comparing tier outputs on matched query sets has
> been performed."

Filling that gap is Stage 1–2. But once the measurement was in place, three of
the results turned out to be about **evaluation practice rather than about this
one system**, and that is where the research now sits: the negative result is
specific to one router, while the protocol, the cost-accounting correction, and
the nondeterminism finding apply to anyone reporting routing savings.

---

## 2. What this contributes

**Four methodological pieces**, which together are the audit protocol:

1. **Static single-tier baselines as the bar.** "Always Tier 1" and "always
   Tier 2" are computed on the identical item set. A router that loses to a
   constant has not earned its complexity, and no amount of savings versus
   always-frontier changes that.
2. **Pre-registration with one-shot frozen test sets.** Success criteria,
   threshold-selection rules and analysis choices were committed to git *before*
   the data existed (`router_v2/PREREGISTRATION.md`,
   `stage7_10/prereg_stage7.md` — check them against `git log`), and each frozen
   test set was scored exactly once. This is what makes a negative result
   credible rather than an artifact of stopping when the numbers looked good.
3. **Knapsack frontiers that separate two different gaps.** Solving the routing
   problem as a multiple-choice knapsack splits the shortfall into a
   *calibration gap* (the router's probabilities are miscalibrated — the
   router's fault, fixable) and a *discreteness gap* (the cost of committing one
   tier per item — the problem's property, not fixable). Reporting one number
   conflates a fixable failure with an inherent limit.
4. **A variance decomposition of the evaluation itself.** Temperature 0 is not
   deterministic. Part of every reported confidence interval is regeneration
   noise that more benchmark items would never reduce, and that share must be
   measured rather than assumed away.

**One portable technical result:**

5. **The cost accounting the routing literature uses is biased, and the bias is
   exactly characterisable.** The confusion-matrix formula for expected energy
   regret is correct only if per-item cost is uncorrelated with routing
   decisions — but routers escalate precisely the items that generate more
   tokens, so it never holds. The exact correction is

   ```
   R_true = R_naive + Σ_{i≠j} Cov( 1{t*=i, t̂=j},  e_j(x) − e_i(x) )
   ```

   On our data this term is **1.9× the size of the regret being reported and
   flips its sign** (−0.250 → +0.284 J/item, reconciling to 2×10⁻¹⁶). It is not
   a deep result — it is algebra from the definition of covariance — but it is
   load-bearing, and it **reproduces on RouteLLM's own released GSM8K data**,
   where naive accounting understates true cost (overstates savings) by up to
   6.3%. See `stage7_10/regret_correction_derivation.md` and
   `stage7_10/external_generalization.md`.

We claim **no new model, architecture or algorithm**, and we do not claim that
learned routing cannot work — one family of zero-API-cost routers was tested on
one workload. The claim is about how savings are currently reported.

---

## 3. What was found

### The router loses to a constant

364 items (HumanEval 164, MMLU 100, GSM8K 100), all three tiers, identical set:

| Routing policy | Accuracy [95% CI] | Energy | vs always-frontier |
|---|---|---|---|
| **EcoLogic (the real classifier)** | **86.8%** [82.9%, 89.9%] | **689 J** | **0.114x** |
| Always Tier 2 (looks at nothing) | **92.3%** [89.1%, 94.6%] | **394 J** | 0.065x |
| Always Tier 1 (looks at nothing) | 87.6% [83.9%, 90.6%] | 677 J | 0.112x |
| Random tier assignment | 90.7% [87.2%, 93.2%] | 2,556 J | 0.421x |
| Always-frontier (`gpt-4o`) | 91.5% [88.2%, 93.9%] | 6,066 J | 1.000x |
| Oracle (cheapest tier that was right) | 96.4% [94.0%, 97.9%] | 386 J | 0.064x |

Read the first two rows together. Sending every query to the middle tier is
**5.5 pp more accurate and 43% cheaper** than the router — so on this workload,
looking at the query is worse than not looking at it. Against always-frontier the
router does save 88.6% of energy, which is the number such systems normally
report, and it is true and nearly meaningless.

The oracle row locates the headroom: a perfect router on these *same three tiers*
would be 9.6 pp more accurate and 44% cheaper. The tier design is fine. The
routing decisions are the problem.

### It is not just that the classifier is keyword-based

A learned replacement was built properly — TF-IDF and local-embedding variants,
logistic regression, threshold calibrated on a held-out split, variant chosen by
train-only cross-validation, all pre-registered, tested once. It scored **92.0%
versus Always-Tier-2's 92.3%**: no significant difference (McNemar p = 1).
Scaling its training data 4.2× and de-noising labels via k=3 majority voting
moved it to 1.10 pp *ahead*, but not significantly (p = 0.22) and at 28% *more*
energy — the pre-registered verdict is **"partial support, inconclusive."**

Why it doesn't work is the more useful finding: held-out per-tier discrimination
is only AUC ≈ 0.67, and 4.2× more data barely moved it (0.656 → 0.665). The
calibration gap to the knapsack frontier closed just 0.83 pp while the
discreteness gap stayed at **0.00 pp**. So the shortfall is not the cost of
one-tier-per-item, and not a shortage of data — **per-item tier success is only
weakly predictable from the prompt text.** That is a property of the workload,
which is why it matters beyond this codebase.

### Temperature 0 is not deterministic

Re-running identical prompts at temperature 0 changed the **graded verdict** on
19.8% of items for Tier 1, versus 1.4% for `gpt-4o`. For that tier, **47% of its
confidence-interval width is regeneration noise** that more items would not
reduce. Consequently the Stage 7 router's 1.10 pp margin falls *inside* its own
2.70 pp sampling+generation interval — two independent routes (McNemar and this
decomposition) agree the comparison is unresolved.

### Two anomalies that break assumptions the design depends on

- **The frontier tier is not the quality ceiling.** `gpt-4o` scored *lowest of
  the three tiers* on code (86.0% MBPP), so always-frontier (89.8%) trails
  Always-Tier-2 (92.6%). Every "quality given up versus the frontier" framing
  inherits this.
- **The classifier is barely stable under prompt formatting.** It agrees with
  itself on only **53.0%** of items between the raw user query and the wrapped
  prompt actually sent to the model, and the wrapped form escalates all 164 code
  items, multiplying energy ~5×. All reported routing behaviour uses the raw
  query — the choice more favourable to the router.

---

## 4. Where to read what

Each report stands alone and states its own limitations.

| Read this | For |
|---|---|
| **`results_report.md`** | **The main report.** Policy comparison, per-tier/per-benchmark accuracy, oracle gap, energy table with sensitivity band, "what failed", limitations. |
| `stage7_10/regret_correction_derivation.md` | Contribution 5: the proposition, proof, and reconciliation to 2×10⁻¹⁶. |
| `stage7_10/external_generalization.md` | Whether contribution 5 affects published work. Sign flip reproduced on RouteLLM's data. |
| `router_v2/README.md` | The learned router: pre-registered, one-shot tested. |
| `stage7_10/SUMMARY.md` | The scaled retest and the statistical hardening. |
| `stage7_10/evaluation_card.md` | The whole evaluation as one document, in Model/Data Cards spirit — measured vs modelled, intended vs unintended use. |
| `stage7_10/contribution_framing.md` | §2 above condensed to a single paragraph, plus notes on what the framing deliberately concedes. |
| `quality_benchmark_report.md` | The first 24-question pilot. **Superseded**; kept for provenance. |

### Directory map

| Path | Contents |
|---|---|
| `benchmark/` | Pipeline: build item sets, call APIs, grade, run the real classifier, analyze. |
| `raw_results/` | Every prompt, response, token count and grade from the main run, plus `analysis.json` and `tables.md`. |
| `router_v2/` | Learned router: pre-registration, pools, R1-vs-R2 ablation, threshold sweep, knapsack frontier, one-shot results. |
| `stage7_10/` | Scaled retest (48,276 calls), the correction derivation, the external check, evaluation card and reproducibility manifest. |

### If you have five minutes

Read the table in §3, then `results_report.md`. That table is the result;
everything else supports it or tries to overturn it.

---

## 5. How the numbers were produced

Every figure comes from real calls to real endpoints. Nothing is estimated,
extrapolated or simulated. Total spend **$46.26**.

**Models.** The audited system's Tier 1/2 slugs (Gemma 3N E4B, Apriel 1.6 15B)
were retired from Together AI before this work, so energy-adjacent substitutes
were used — documented rather than swapped silently:

| Tier | Model | Provider | Assumed rate (J/1k tok) |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B` | Together AI | 0.5 |
| 2 | `openai/gpt-oss-20b` | Together AI | 1.5 |
| 3 | `gpt-4o` | OpenAI | 60.0 |

**Benchmarks**, chosen so grading is objective rather than judged by another
model: HumanEval (executed against the official test suites), MMLU (letter
match), GSM8K (final-answer match). MBPP substitutes for HumanEval in training
pools, because all 164 HumanEval items are in the frozen test set and reusing
them for training would be contamination.

**Statistics.** 95% Wilson intervals on every accuracy figure; McNemar's exact
test for paired comparisons on identical items; ±5× per-tier sensitivity sweeps
on every energy conclusion.

---

## 6. What this does NOT establish

The reports are only useful if their bounds are clear.

- **No joule was measured.** Energy is token counts — which are real — times the
  audited system's own assumed J/1k-token rates. No wattmeter, no GPU telemetry.
  This is the single weakest point in the work. Every energy claim therefore
  carries a ±5× sensitivity band, and contributions 1–5 are better understood as
  being about **cost accounting under per-item cost dispersion**, which is
  rate-model independent, than about energy specifically.
- **Benchmarks are not production traffic.** Single-turn, self-contained,
  auto-gradable academic tasks. Transfer to real usage is not measured.
- **Tiers 1 and 2 are substitutes.** The routing *logic* is audited; the
  deployed model stack is not.
- **"Routing does not work" is not a claim this can support.** One system, one
  substituted model stack, four benchmarks, one family of zero-API-cost routers.
  RouteLLM and FrugalGPT report gains in their own settings.

---

## 7. Reproducing it

`requirements.txt` covers only the app; the evaluation has its own pinned set.

```bash
pip install -r requirements-eval.txt   # NOT requirements.txt
export TOGETHER_API_KEY=...            # Tiers 1-2
export OPENAI_API_KEY=...              # Tier 3

python3 benchmark/build_benchmark.py   # fixed seed -> identical item set
python3 benchmark/run_benchmark.py     # ~1,100 calls; resumable
python3 benchmark/grade.py             # executes HumanEval in a subprocess sandbox
python3 benchmark/router.py            # runs the real classifier from backend/main.py
python3 benchmark/analyze.py           # writes raw_results/analysis.json + tables.md
```

Re-scoring a **routing** change costs nothing, because it reuses cached
generations:

```bash
python3 benchmark/router.py && python3 benchmark/analyze.py   # no API calls
```

Every script is resumable and keyed on `(tier, item_id)`, so an interrupted run
never repeats or double-bills a completed call.
`stage7_10/reproducibility_manifest.md` lists every seed, package version, API
endpoint and model version across all stages — and flags which are **not**
pinnable, notably that no provider exposes an immutable model revision, so exact
regeneration is guaranteed by nothing under our control.

Stage 7's raw generations are 635 MB and are **not** committed. What is committed
instead: per-sample grades, tokens and costs for all 48,276 calls
(`stage7_10/s7_*_samples.csv.gz`), plus full response text for the cases where
grading is most contestable — k=3 disagreements, ungradable output, truncations,
and a random control sample (`stage7_10/s7_*_audit_sample.jsonl.gz`). Any
individual grading decision can be audited; the full corpus cannot be rebuilt
from this repository.
