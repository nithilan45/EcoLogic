# Stages 11–13 — the decomposition, RouterBench at scale, and a refuted hypothesis

> **New here?** Read [`../EVALUATION.md`](../EVALUATION.md) for the research goal
> and [`../paper/`](../paper/) for the write-up. **This group's role:** turn
> Stages 1–10 from "measure routers against the right baseline" into an
> *explanation* — a decomposition that says which of three possible causes makes
> a routing gain small — and then test that explanation on 100× more data than
> our own \$48 bought.

Nothing in `raw_results/`, `results_report.md`, `router_v2/` or `stage7_10/` was
modified. Everything here is new, in `stage11_13/`, `external_data/` and
`paper/`.

---

## The one-paragraph version

A router's gain over the correct baseline factors as
**complementarity × predictability − estimation error**. On RouterBench
(36,494 prompts × 11 models, all 55 pairs × 8 benchmark families)
**complementarity is abundant — 12.11 pp of oracle headroom at matched cost —
and the best of four router families captures 0.59 pp of it, 4.6%.** We
pre-registered the explanation that predictability is the missing factor, and
**the data refuted it**: using k=3 replicate generations of 5,000 prompts
(the one thing no public routing dataset has), per-item difficulty turns out to
be highly *reliable* — 77–98% of outcome variance is stable between-item signal
— implying a Bayes-optimal AUC of 0.95–0.99. So the information is there and the
binding term is the **generalisation gap**: which model will succeed is a stable
property of the item that routers cannot read off the prompt. That gap survives
nine router families up to a prompted 70B model and an **end-to-end fine-tuned**
encoder, and a learning curve whose power-law asymptote is 0.74 AUC. Under a
paired bootstrap with Holm correction, **no** router family's AUC advantage over
a frozen MiniLM + logistic baseline is distinguishable from zero. The whole
analysis **replicates** on RouterBench's independent 5-shot release
(κ = 11.44 pp, realised ρ = 5.5%).

---

## Status at a glance

| Stage | What it was for | Status |
|---|---|---|
| **11** | Formalise the decomposition; prove it; validate it; measure it on RouterBench | **Complete.** 14/14 propositions validated against brute-force LP/Monte-Carlo. H1 **supported**, H2 **supported**, on both the 0-shot and the 5-shot release. |
| **12** | Router-free predictability ceiling from replicate generations | **Complete, and it refuted our own H3.** Reported as such. |
| **12b** | Learning curves — would more data close the gap? | **Complete. Exploratory, not pre-registered** (`DEVIATIONS.md` D3). |
| **13** | Router-strength ladder: prompted LLM router, fine-tuned LLM router | **Partly blocked.** R-b (prompted 70B) complete. **R-c, the fine-tuned generative LLM, is BLOCKED** — it trained, then proved unservable through four routes, and re-training on a servable base was refused for insufficient balance (`DEVIATIONS.md` D8, `s13_ftblocked.json`). |
| **13b** | The substitute for R-c: **unfreeze the encoder** and fine-tune it end-to-end | **Complete. Exploratory, not pre-registered** (`DEVIATIONS.md` D7). C1 **not met**. |
| **13c** | The same, on RouterBench's 29k training items | **Running.** Exploratory. No result is quoted anywhere until it lands; nothing else in this group depends on it. |
| **Paper** | Workshop-length draft with proofs and appendix | **Complete.** `../paper/main.tex`, compiles with `pdflatex` to 12 pages (~6.3 body + references + appendix). |

Additional spend for these stages: **$9.71** against a pre-registered gate of
**$25**. The gate was never the binding constraint — the provider's account
balance was. Project total **$57.80**, against the original $150 envelope.
RouterBench (both releases), the decomposition, the learning curves, the
end-to-end encoder rungs and all numerical validation cost **nothing**: they
analyse already-released outcomes, or run on CPU.

---

## Stage 11 — the decomposition

### The claim

For a workload and a model set, let `S(b)` be the best expected utility at
expected cost `b` from any **query-independent** policy, `A†(b)` from any
**router** (any function of the query), and `A*(b)` from an **oracle** that sees
realised outcomes. Then `S(b) ≤ A†(b) ≤ A*(b)`, and with

```
complementarity  kappa(b) = A*(b) - S(b)
predictability   rho(b)   = (A†(b) - S(b)) / kappa(b)
```

any fitted router realises `rho(b)·kappa(b) − eps`. The three terms blame three
different things: `kappa` the **model set**, `rho` the **task**, `eps` the
**engineer**.

Five propositions, all proved in `theory.md` and all checked numerically:

| | Result | Why it matters |
|---|---|---|
| P1 | the ladder, and concavity | the frame |
| P1b | **`rho = 1` exactly when outcomes are deterministic given `x`** | every unit of `rho < 1` is outcome noise, so a single-generation oracle overstates attainable headroom |
| P2 | two-model closed forms; `rho` = ratio of tail dispersion of the conditional mean to that of the realisation; `min(p01,p10)` for the binary unconstrained case | makes both factors computable |
| P3 | **`gain ≤ sqrt(beta(1-beta))·sd(delta)` for every router**, by Cauchy–Schwarz | a router-free cap — and *the same covariance object* as the Stage 8 cost correction |
| P4 | `Var(delta)` is identified from k ≥ 2 replicates | the reason our own data is not redundant with RouterBench |
| P5 | `AUC* = 1/2 + E|eta(X)−eta(X')| / (4a(1−a))` | converts replicate variance into a prediction about a number four learners already measured |

`s11_validate.py` → `s11_validate.json`: **14/14 checks pass**, comparing each
closed form against `scipy` HiGHS LP solutions or Monte Carlo. The suite also
shows the *uncorrected* variance estimator inflates `Var(delta)` by **2.33×**,
which is the size of the mistake a single-generation dataset forces.

### Measured on RouterBench

`withmartian/routerbench`, 0-shot file (SHA-256 pinned in the result JSON):
36,494 prompts × 11 models = 401,434 outcomes, per-item graded score and
per-item dollar cost. Families with n ≥ 100: MMLU 14,042, HellaSwag 10,042,
GSM8K 7,450, ARC 1,470, Winogrande 1,267, Chinese 785, other 931, MBPP 427.

`S(b)` is computed exactly (upper concave envelope); `A*(b)` exactly (Lagrangian
solution of the multiple-choice knapsack LP on **per-item** costs). Routers
commit using their own predictions and per-model *mean* costs — a deployed
router cannot know an item's token count in advance — but are charged
**realised** per-item cost, which is the Stage 8 correction applied.

At `beta = 0.5` (where P3's cap is largest, so the most routing-favourable
point), medians over 440 pair × family cells with 95% bootstrap intervals:

| Router | out-of-fold AUC | gain (pp) | realised `rho` | cells with gain > 0 |
|---|---|---|---|---|
| TF-IDF + logistic | **0.7160** | **+0.585** [0.442, 0.777] | **0.046** [0.036, 0.059] | 311/440 |
| MiniLM + logistic | 0.7081 | +0.455 [0.286, 0.609] | 0.034 | 288/440 |
| MiniLM + boosted trees | 0.6772 | +0.402 [0.268, 0.544] | 0.033 | 290/440 |
| MiniLM + MLP (256, 64) | 0.6626 | +0.226 [0.138, 0.326] | 0.017 | 269/440 |
| **complementarity `kappa`** | | **12.11 pp** [11.15, 12.92] | | |

**H1 (complementarity ≥ 5 pp): SUPPORTED.** **H2 (realised `rho` ≤ 30%):
SUPPORTED**, at 4.6%.

This is **not** a null result. 49 of 55 pairs show a positive mean gain, 28 have
raw p < 0.05, **26 survive Benjamini–Hochberg** at FDR 0.05 and **13 survive
Holm–Bonferroni** at family-wise α = 0.05. Routing works; it recovers about a
twentieth of what is available. Reading gains against always-frontier hides a
factor of roughly twenty.

Note the MLP is the **worst** of the four despite the most capacity — the same
inversion Stage 7c found in-house, and the first sign that model class is not
the binding constraint.

### It replicates on the 5-shot release

RouterBench ships a second, independent set of generations in which every prompt
carries five in-context exemplars. It was pre-registered as a replication and
analysed with identical code:

| Quantity | 0-shot (primary) | 5-shot (replication) |
|---|---|---|
| items analysed | 36,494 | 36,480 (28 dropped for missing cells, `DEVIATIONS.md` D6) |
| complementarity `kappa` | 12.11 pp [11.15, 12.92] | 11.44 pp [10.66, 12.27] |
| best router | TF-IDF + logistic | TF-IDF + logistic |
| its out-of-fold AUC | 0.7160 | 0.7172 |
| its realised gain | +0.585 pp [0.442, 0.767] | +0.640 pp [0.519, 0.790] |
| its realised `rho` | 4.6% [3.6, 5.9] | 5.5% [4.4, 6.8] |
| pairs with positive gain | 49/55 | 50/55 |
| pairs surviving BH / Holm | 26 / 13 | 22 / 11 |
| H1, H2 | supported | supported |

---

## Stage 12 — the hypothesis we refuted

We pre-registered H3: that the router-free bound would be ≤ 50% of `kappa`, so
that at least half the measured complementarity would be provably unreachable.
The falsification rule was written down in advance, including the sentence "then
the ceiling argument is vacuous and we must retreat."

**It is vacuous.** At `beta = 0.5` the P3 bound evaluates to **8.4–15.6 pp**
against a single-draw `kappa` of **3.3–9.5 pp** — a median ratio of **2.82**.
The inequality is correct and is attained on a two-point distribution
(`s11_validate.py` checks both), but real `delta` is diffuse and Cauchy–Schwarz
loses exactly that slack.

The reason is more interesting than the failure. Using k = 3 replicates
(5,000 pool items at temperature 0.7 = 45,000 calls; 364 test items at
temperature 0 = 3,276 calls), **per-item difficulty is highly reliable**:

| Set | n | Tier 1 (9B) | Tier 2 (20B) | Tier 3 (GPT-4o) |
|---|---|---|---|---|
| reliability, pool, T=0.7 | 5,000 | 0.895 | 0.873 | 0.942 |
| reliability, test, T=0.0 | 364 | 0.772 | 0.902 | 0.983 |
| implied `AUC*` (P5, Beta fit), pool | | 0.992 | 0.987 | 0.997 |

*Reliability* is the share of per-item outcome variance that is stable
between-item signal rather than regeneration noise. So whether a model answers an
item correctly is **not a coin flip** — and the pre-registered §3.4 consistency
check, which predicted the replicate-implied `AUC*` would land within 0.05 of the
measured 0.65–0.68 plateau, **came out 0.99 against 0.68 and is recorded as
INCONSISTENT**. That check existed precisely because it could embarrass us.

**Consequence:** `rho ≈ 1`, predictability is not the bottleneck, and the whole
gap is `eps`. The claim changes from "per-item success is unpredictable" to
"**per-item success is highly reliable yet only weakly inferable from prompt
text**." Full accounting in `DEVIATIONS.md` D1.

Two findings do survive:

- **A peeking policy captures only 34–55% of the single-draw `kappa`.** A policy
  allowed to see one graded outcome per model — information no deployed router
  has — realises 1.24/1.95/2.69 pp against `kappa` of 3.68/3.52/4.87 pp, when
  evaluated on held-out replicates. This is *not* a bound on `A†`
  (`DEVIATIONS.md` D2 corrects a pre-registration error on exactly this point),
  but it says how much reported oracle headroom survives when you cannot see the
  draw you are scored on.
- **8–35% of the variance of the observed advantage is generation noise**, and
  the share scales with output length: 8% for terse GPT-4o, 35% for the
  long-reasoning 9B tier that changes its graded verdict on 19.8% of items
  between identical temperature-0 calls.

---

## Stage 12b — would more data close it? (exploratory)

Fixed 20% held-out set on RouterBench; training subsets from 250 to 29,000
items, repeats at the small sizes; power-law fit `AUC(n) = A − B·n^(−c)`.

| Router | AUC at 29k | asymptote `A` | AUC(10⁶) | AUC(10⁹) | doublings per +0.01 AUC |
|---|---|---|---|---|---|
| MiniLM + logistic | 0.7081 | **0.741** | 0.726 | 0.738 | 2.5 |
| MiniLM + MLP | 0.6612 | *fit hit the bound at 1.0; unreliable* | 0.717 | 0.809 | — |

Realised gain **does** improve with data (+1.15 pp at 8k → +2.10 pp at 29k,
realised `rho` = 0.105), so this is not "data doesn't help." But the extrapolated
ceiling stays far below what capturing 12 pp of `kappa` would require. The MLP's
power-law fit is degenerate (the asymptote parameter hit its upper bound) and is
reported as unreliable rather than quoted.

---

## Stage 13 — the router-strength ladder

Same 1,500-item Stage 7 CALIBRATION split as Stage 7c, so the comparison is
apples to apples. The frozen Stage 7 test set was not touched.

`ΔAUC` is a 2,000-resample paired bootstrap over items against the frozen
MiniLM + logistic reference refitted on the same TRAIN split, Holm-corrected
across the four new rungs. `gain` is the realised matched-cost gain at
`beta = 0.5` against `kappa = 4.07 pp` on this split.

| Router | Representation | mean AUC | ΔAUC [95%] | p (Holm) | gain (pp) |
|---|---|---|---|---|---|
| k-NN (k=50) | MiniLM (frozen) | 0.6521 | — | — | — |
| Gradient boosting | MiniLM (frozen) | 0.6547 | — | — | — |
| Random forest (**train AUC 0.9999**) | MiniLM (frozen) | 0.6703 | — | — | — |
| Logistic regression, Stage 7c | MiniLM (frozen) | 0.6816 | — | — | — |
| Logistic regression, refit here | MiniLM (frozen) | 0.6896 | *reference* | — | **+0.556** |
| Prompted LLM, zero-shot | Llama-3.3-70B-Instruct | 0.6416 | −0.048 [−0.097, +0.004] | 0.256 | −0.444 |
| Prompted LLM, 4-shot | Llama-3.3-70B-Instruct | **0.7105** | +0.021 [−0.019, +0.062] | 0.954 | **−1.311** |
| Fine-tuned end-to-end | MiniLM-L6 (**unfrozen**) | 0.6945 | +0.005 [−0.014, +0.025] | 1.000 | +0.533 |
| Fine-tuned end-to-end | MiniLM-L12 (**unfrozen**) | 0.6912 | +0.002 [−0.018, +0.023] | 1.000 | +0.533 |
| LoRA fine-tune, 3 epochs | Gemma-3-27B-it | **BLOCKED** | — | — | — |

**Pre-registered criterion C1** — a stronger router must beat Stage 7c's best
(0.6816) by ≥ 0.05 for "you tested a weak router" to become live again — is
**not met**: the largest point estimate is +0.029. **C2** (realised gain ≥ 1.0 pp
over the frozen reference) is **not met** either: the best new rung is *0.02 pp
worse* than the reference. And no rung's ΔAUC is distinguishable from zero.

Three things in that table are worth more than the verdict.

1. **Unfreezing the encoder is the honest answer to "you used frozen
   features"**, and it buys +0.005 [−0.014, +0.025]. Its *inner-validation* AUC
   reaches 0.751 while held-out is 0.6945 — the capacity to fit routing labels
   exists and does not transfer. That is `eps` made visible.
2. **The highest-AUC rung has the worst matched-cost gain.** The prompted 4-shot
   router ranks best by AUC (0.7105) and is *worse than ignoring the query*
   (−1.31 pp) at the same budget on the same items. AUC is computed per model
   and is invariant to monotone rescaling of that model's scores, but a
   matched-cost policy ranks items by the *difference* of two models' predicted
   utilities. A router can order items correctly within each model and get every
   cross-model comparison wrong. The routing literature reports per-model AUC
   almost universally; here the standard metric and the deployable quantity
   **disagree in sign**.
3. **The random forest is the diagnostic:** training AUC 0.9999 (perfect
   memorisation) and held-out AUC *below* logistic regression. Ample capacity,
   no generalisation — the signature of a target that is not a smooth function
   of the input representation, not of an inadequate model class.

**What is blocked, and what it costs.** The pre-registered top rung was a
fine-tuned *generative* LLM. It trained without incident ($6.71) and then proved
unservable through four routes, each probed rather than assumed: no serverless
LoRA on the account; dedicated-endpoints v1 creation retired platform-wide; no
certified v2 serving config for `gemma-3-27b-it`; and re-training on a v2-servable
base (`Qwen/Qwen3.5-9B`, which is also the Tier-1 model here) refused with
`402 insufficient_balance`. Full errors in `s13_ftblocked.json` and
`DEVIATIONS.md` D8. Stated plainly: **a reader who believes a fine-tuned
generative LLM router would clear +0.05 has not been answered by one.** They have
been answered by a fine-tuned encoder, a prompted 70B model, a memorising random
forest and a learning curve.

**Cost.** Pre-registered gate $25, enforced in code (`s13_spend.json`; every call
books the provider's own reported usage and the run aborts at the gate). Actuals:
prompted zero-shot $0.58, prompted 4-shot $2.41, LoRA training $6.71, fine-tuned
inference $0.00 (blocked), end-to-end encoders $0.00 (CPU). **Total $9.71**
against a projection of ~$16. The gate never bound; the provider's balance did.

---

## Files

| File | Contents |
|---|---|
| `prereg_stage11_13.md` | **Pre-registration, committed before any result artifact** — hypotheses, falsification rules, family map, router families, statistics, cost gate |
| `DEVIATIONS.md` | Eight deviations, including **D1 (H3 refuted)**, **D2 (a pre-registration statement that was wrong)** and **D8 (the blocked rung)** |
| `theory.md` | The decomposition, all five propositions with proofs, estimation, and what the theory does not show |
| `decomp.py` | The computational core: exact `S(b)`, exact `A*(b)`, router value at matched cost, the P3 ceiling, replicate variance components, Bayes AUC, Holm and BH |
| `s11_validate.py` / `.json` | **14/14 proposition checks** against brute-force LP and Monte-Carlo references |
| `s11_routerbench.py` / `_{0,5}shot.json` / `_pairs_{0,5}shot.csv.gz` | The decomposition on both RouterBench releases: 55 pairs × 8 families × 9 operating points |
| `s12_ceiling.py` / `s12_ceiling.json` | Replicate variance components, reliability, the vacuous ceiling, the peeking policy, the P5 check |
| `s12b_learning_curve.py` / `.json` / `.csv` | Learning curves and power-law extrapolation (exploratory) |
| `s13_llm_router.py` / `.json` / `s13_spend.json` / `s13_state*.json` | Prompted and fine-tuned LLM routers, the enforced cost gate, the full ledger, and the paired-bootstrap C1/C2 verdicts |
| `s13_ftblocked.json`, `s13_endpoint_probe.json` | The four probed routes by which the fine-tune could not be served, with verbatim provider errors |
| `s13b_encoder_finetune.py` / `.json` / `s13b_encoder_*.jsonl` | End-to-end fine-tuned encoders on the in-house split (exploratory, $0, CPU) |
| `s13c_encoder_routerbench.py` / `.json` | The same on RouterBench's 29k training items, against frozen routers refit on identical items |
| `s13_prompted_{0,4}shot.jsonl` | Per-(item, tier) router scores, resumable and keyed |
| `s13_ft_train.jsonl` | The exact 10,500-example fine-tuning file, for reproduction |

### What is deliberately *not* in the repo

The RouterBench pickles (99 MB and 171 MB) and the derived MiniLM embedding cache
(53 MB) are gitignored. Both are regenerated by the commands in
`../paper/README.md`, and the pickle is pinned by SHA-256 in
`s11_routerbench_0shot.json` so a reader can verify they analysed the same bytes.
