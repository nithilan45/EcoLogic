# EcoLogic — independent quality & energy evaluation

**Start here if you have just been added to this repository and want to know
what was measured, what was found, and where the numbers live.**

Alongside the product, this repository holds an evaluation of EcoLogic's routing
claims, in `benchmark/`, `raw_results/`, `router_v2/` and `stage7_10/`. It reads
the classifier out of `backend/main.py` but changes no product code, so nothing
here affects how the deployed app behaves.

---

## 1. Why this exists

EcoLogic routes each query to one of three model tiers, escalating only when the
query looks hard, and reports the energy saved versus sending everything to a
frontier model. The project's own limitations section conceded:

> "no formal quality evaluation comparing tier outputs on matched query sets has
> been performed."

That is the gap this work fills. The question is not "does the router run" but
**does routing actually buy the energy/quality trade-off it claims** — measured
on identical query sets, with real API calls, objective grading, and confidence
intervals.

Everything reported was produced by real calls to real endpoints. No number here
is estimated, extrapolated, or simulated. Total spend: **$46.26**.

---

## 2. The five findings, in order of importance

**1. Routing works, but the router is the weak link, not the tier design.**
On 364 benchmark items, EcoLogic's keyword classifier reached 86.8% accuracy
using 11.4% of always-frontier energy. But the *oracle* router — the cheapest
tier that actually got each item right — reached 96.4% at 6.4% of frontier
energy. So a perfect router on these same three tiers would have been **both
9.6 pp more accurate and 44% cheaper**. Almost all the headroom is in routing
decisions, not in the models.

**2. A static policy beats the router.** "Send everything to Tier 2" scored
92.3% at 6.5% of frontier energy — more accurate *and* cheaper than the
classifier's 86.8% at 11.4%. The routing logic is not earning its complexity on
this workload.

**3. Replacing the keyword classifier with a trained one did not fix it.**
A learned router (TF-IDF and embedding variants, logistic regression, threshold
calibrated on held-out data, pre-registered before running) scored 92.0% versus
Always-Tier-2's 92.3% — no significant difference. Scaling its training data
4.2x and denoising the labels moved it to 1.10 pp *ahead*, but not significantly
(p = 0.22) and at 28% more energy. Verdict: **partial support, inconclusive**.
The remaining gap to the theoretical frontier looks structural — per-item tier
success is only weakly predictable from the prompt text (held-out AUC ~0.67).

**4. "Temperature 0" is not deterministic, and it matters for evaluation.**
Re-running identical prompts at temperature 0 changed the *graded verdict* on
19.8% of items for Tier 1, versus 1.4% for `gpt-4o`. For the long-reasoning
tier, 47% of its confidence-interval width is regeneration noise that more
benchmark items would not reduce. Any single-run accuracy comparison of these
tiers is reproducible only to within a few points.

**5. A measurement bug that likely affects published routing work.** The
standard confusion-matrix formula for energy regret gives the **wrong sign**
whenever per-item cost correlates with routing decisions — which it always does,
since routers escalate exactly the items that generate more tokens. We derive
the exact correction term, verify it reconciles to numerical precision, and
reproduce the sign flip on RouteLLM's own released data. See
`stage7_10/regret_correction_derivation.md`.

### Two anomalies worth knowing before you read anything else

- **`gpt-4o` is not the quality ceiling the design assumes.** It scored *lowest
  of the three tiers* on code (86.0% on MBPP), so "always-frontier" trails
  Always-Tier-2 (89.8% vs 92.6%). Every "quality given up versus the frontier"
  framing inherits this.
- **The classifier is highly prompt-sensitive.** It agrees with itself on only
  **53.0%** of items between the raw user query and the wrapped prompt actually
  sent to the model — and the wrapped form escalates all 164 code items,
  multiplying energy ~5x. All reported routing behaviour uses the raw query,
  which is the choice more favourable to EcoLogic.

---

## 3. Where to read what

Read in this order. Each report stands alone and states its own limitations.

| Read this | For |
|---|---|
| **`results_report.md`** | **The main report.** Four-policy comparison, per-tier/per-benchmark accuracy, oracle gap, energy table with sensitivity band, "what failed" section, limitations. Start here. |
| `router_v2/README.md` | The learned router: can a trained classifier beat the keyword one? Pre-registered, one-shot tested. |
| `stage7_10/SUMMARY.md` | Whether more/cleaner training data closes the gap (it mostly doesn't), the regret-formula correction, the external check, and the statistical hardening. |
| `stage7_10/evaluation_card.md` | A single-document summary of the whole evaluation in the spirit of Model/Data Cards — what is measured vs modelled, and intended/unintended use. |
| `quality_benchmark_report.md` | The first, smaller pilot (24 hand-written questions, LLM judge). **Superseded**; kept for provenance. |

### Directory map

| Path | Contents |
|---|---|
| `benchmark/` | The pipeline: build item sets, call the APIs, grade, run the real classifier, analyze. |
| `raw_results/` | Every prompt, response, token count and grade from the main run, plus `analysis.json` and `tables.md`. |
| `router_v2/` | Learned router: pre-registration, training pool, R1-vs-R2 ablation, threshold sweep, MCKP frontier, one-shot test results, limitations. |
| `stage7_10/` | Scaled retest (48,276 calls), the regret-correction derivation, the RouteLLM external check, and the evaluation card / reproducibility manifest. |

### If you only have five minutes

Read §2 above, then the four-policy table in `results_report.md`. That table is
the result; everything else either supports it or tries to overturn it.

---

## 4. How the numbers were produced

**Models under test.** The paper's Tier 1/Tier 2 slugs (Gemma 3N E4B, Apriel
1.6 15B) were retired from Together AI before this evaluation, so
energy-adjacent substitutes were used, with the change documented rather than
made silently:

| Tier | Model | Provider | Paper rate (J/1k tok) |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B` | Together AI | 0.5 |
| 2 | `openai/gpt-oss-20b` | Together AI | 1.5 |
| 3 | `gpt-4o` | OpenAI | 60.0 |

**Benchmarks**, chosen so grading is objective rather than judged by another
model: HumanEval (164 items, graded by executing the official test suites),
MMLU (100 items across 10 subjects, letter match), GSM8K (100 items, final-answer
match). MBPP substitutes for HumanEval in training pools, since all 164
HumanEval items are in the frozen test set and reusing them for training would
be contamination.

**Statistics.** 95% Wilson intervals on every accuracy figure; McNemar's exact
test for paired comparisons on identical items; ±5x per-tier sensitivity sweeps
on every energy conclusion.

**Discipline.** Every hypothesis test in `router_v2/` and `stage7_10/` was
pre-registered — success criteria, threshold-selection rule and analysis choices
committed to git *before* the data existed — and each frozen test set was
evaluated exactly once. `router_v2/PREREGISTRATION.md` and
`stage7_10/prereg_stage7.md` are those commitments; you can check them against
git history.

---

## 5. What this does NOT establish

Stated plainly, because the reports are only useful if their bounds are clear.

- **Energy is modelled, never measured.** All energy figures are token counts
  multiplied by the paper's own J/1k-token rates. No wattmeter, no GPU telemetry.
  Token counts are real; joules are arithmetic on an assumption. This is the
  single weakest point in the whole evaluation, which is why every energy claim
  carries a ±5x sensitivity band.
- **Benchmarks are not EcoLogic's traffic.** Single-turn, self-contained,
  auto-gradable academic tasks. Conclusions transfer to real usage only as far as
  that resemblance holds, which is not measured.
- **Tiers 1 and 2 are substitutes**, not the paper's models. Routing *logic* is
  audited; the deployed model stack is not.
- **Negative results are about this workload and these routers**, not about
  query routing in general. RouteLLM and FrugalGPT report gains in their own
  settings.

---

## 6. Reproducing it

Note that `requirements.txt` covers only the app; the evaluation has its own
pinned dependency set.

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

Every script is resumable and keyed on `(tier, item_id)`, so an interrupted run
never repeats or double-bills a completed call. `stage7_10/reproducibility_manifest.md`
lists every seed, package version, API endpoint and model version used across all
stages, and flags which ones are **not** pinnable — notably that none of the three
providers expose an immutable model revision, so exact regeneration is not
guaranteed by anything under our control.

Raw generations for the largest run are ~288 MB and are not committed. What is
committed instead: per-sample grades, token counts and costs for all 48,276 calls
(`stage7_10/s7_*_samples.csv.gz`), plus full response text for the cases where
grading is most contestable — disagreements, ungradable output, truncations, and a
random control sample (`stage7_10/s7_*_audit_sample.jsonl.gz`). So any grading
decision can be checked without the full corpus.
