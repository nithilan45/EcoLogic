# Stage 9 — does the correction term matter outside this pipeline?

**Result: yes, and it was checkable.** RouteLLM's release contains enough
item-level material to reconstruct per-item cost for one of its three
benchmarks, and on that benchmark the Stage 8 correction reconciles exactly
(residual 8.7e-19), **flips the sign of measured energy/cost regret at one of
their ten published operating points**, and makes naive cost accounting
overstate savings by up to **6.3%**. The effect is real but considerably milder
than in our own pipeline, for a reason that is itself informative.

## Sources checked, and what each actually contains

`github.com/lm-sys/RouteLLM` at commit `0b64fda` (cloned 2026-09-06), the
codebase for Ong et al., *RouteLLM: Learning to Route LLMs from Preference
Data*, ICLR 2025 (arXiv:2406.18665).

| Released artifact | prompt | per-item correctness | response text | token counts | per-item $ cost | cost reconstructable? |
|---|---|---|---|---|---|---|
| `evals/gsm8k/gsm8k_responses.csv` (1,319 rows) | yes | yes, both models | **yes, both models** | no | no | **yes, by tokenizing** |
| `evals/mmlu/responses/mmlu_*.csv` (57 files) | yes | yes, both models | no | no | no | no |
| `evals/mt_bench/judgements.jsonl` | judge prompt | judge `score` | no model responses | no | no | no |

**No RouteLLM artifact publishes token counts or per-item cost as a numeric
field.** For GSM8K it is nonetheless recoverable, because the full response text
of both routed models is released and can be tokenized with those models' real
tokenizers. That is a reconstruction from their own published outputs, not a
simulation or a stand-in dataset.

For MMLU and MT-Bench the reconstruction is **not possible**: those files ship
only the boolean correctness of each model per item. So the honest scope of this
external check is *one* benchmark, 1,307 items after decontamination — not the
whole RouteLLM evaluation.

### RouteLLM's cost axis is a call fraction, not a token cost

This is worth stating plainly because it is the condition under which the Stage 8
proposition bites. `routellm/evals/evaluate.py` contains **no occurrence of the
strings `cost` or `token`**. Every reported metric — `20%/50%/80% qual`, `AUC`,
`APGR` — is computed against `strong_percentage`, the percentage of queries sent
to the strong model. The README's headline claim, "reduce costs by up to 85%
while maintaining 95% GPT-4 performance", is therefore a statement about call
counts, converted to dollars only by assuming a constant price per call within
each model. That assumption is exactly what the correction term measures.

To be fair to the authors: for a two-model router with a fixed price per call,
call fraction is a reasonable proxy, and they never claim it is token-exact. The
point here is not that RouteLLM is wrong, it is that the proxy has a measurable
bias whose sign is predictable, and nothing in the release lets a reader check it
for two of the three benchmarks.

## Method

Faithful to their code, with no re-tuning:

- **Router**: RouteLLM's own released checkpoint `routellm/bert_gpt4_augmented`
  (XLM-RoBERTa, 3 labels), run locally on CPU. `calculate_strong_win_rate` is
  replicated exactly from `routellm/routers/routers.py` — softmax over logits,
  then `1 - (p[tie] + p[weak])`. Routing follows `controller.py`:
  `win_rate >= threshold` goes to the strong model.
- **Thresholds**: their own grid, `pd.qcut(win_rates, 10, retbins=True)`, the
  same construction `benchmarks.py::GSM8K.evaluate` uses.
- **Decontamination**: their `contaminated_prompts.jsonl`, giving 1,307/1,319
  items — the same count their code prints.
- **Per-item tokens**: `tiktoken` `cl100k_base` for `gpt-4-1106-preview` (its
  actual tokenizer) and the `mistralai/Mixtral-8x7B-Instruct-v0.1` SentencePiece
  tokenizer for the weak model. Prompt and completion tokenized separately.
- **Prices** (published list prices contemporaneous with the paper, $ per 1M
  tokens): GPT-4-1106-preview 10.00 in / 30.00 out; Mixtral-8x7B-Instruct
  0.60 in / 0.60 out. These are an input we supply, not something RouteLLM
  released; the *conclusions below are about the naive-vs-true gap*, which is
  driven by within-model token spread rather than by the absolute price level.
- **Oracle** `t*`: the cheaper model when it is correct, else the strong model
  when only it is correct, else the cheaper model. Mixtral is cheaper than GPT-4
  on every single item (asserted in code), so the oracle is well defined.

## Results, GSM8K (n = 1,307)

Mixtral accuracy 63.73%, GPT-4 accuracy 85.77%. Mean cost per item: Mixtral
$0.000119, GPT-4 $0.004313. Within-model coefficient of variation of per-item
cost: **0.356** (Mixtral), **0.428** (GPT-4).

| Threshold | Strong % | Accuracy % | True $/item | Naive $/item | Naive error | `R_naive` | correction | `R_true` | sign flip |
|---|---|---|---|---|---|---|---|---|---|
| 0.213 | 100.0 | 85.77 | 0.004313 | 0.004313 | 0.00% | +0.0029685 | -0.0001735 | +0.0027950 | |
| 0.421 | 90.0 | 84.47 | 0.003910 | 0.003893 | -0.43% | +0.0025481 | -0.0001566 | +0.0023915 | |
| 0.456 | 80.0 | 81.56 | 0.003498 | 0.003472 | -0.73% | +0.0021277 | -0.0001481 | +0.0019796 | |
| 0.477 | 70.0 | 79.27 | 0.003081 | 0.003055 | -0.85% | +0.0017105 | -0.0001474 | +0.0015631 | |
| 0.498 | 60.0 | 78.19 | 0.002663 | 0.002635 | -1.08% | +0.0012901 | -0.0001448 | +0.0011453 | |
| 0.517 | 50.0 | 76.28 | 0.002257 | 0.002218 | -1.74% | +0.0008729 | -0.0001343 | +0.0007387 | |
| 0.535 | 40.0 | 73.37 | 0.001854 | 0.001797 | -3.05% | +0.0004525 | -0.0001170 | +0.0003355 | |
| 0.555 | 30.0 | 70.93 | 0.001426 | 0.001377 | -3.44% | **+0.0000321** | -0.0001244 | **-0.0000923** | **YES** |
| 0.582 | 20.0 | 69.01 | 0.001006 | 0.000960 | -4.60% | -0.0003851 | -0.0001272 | -0.0005123 | |
| 0.626 | 10.0 | 66.49 | 0.000575 | 0.000539 | **-6.31%** | -0.0008055 | -0.0001372 | -0.0009427 | |
| 0.818 | 0.1 | 63.81 | 0.000122 | 0.000122 | -0.25% | -0.0012227 | -0.0001732 | -0.0013959 | |

Costs in $/item. `R_naive + correction = R_true` at every row; the largest
absolute residual over the whole sweep is **8.7e-19**, i.e. floating-point noise.
The Stage 8 proposition therefore validates on external data as well as on ours.

Three things to take from the table.

**The sign flip reproduces.** At the 30%-strong operating point the naive formula
reports the router spending *more* than the oracle (+0.0000321 $/item) when it
actually spends *less* (-0.0000923 $/item). This is the same failure mode as
Stage 6, on someone else's router, someone else's models and someone else's data.
It happens where true regret is near zero, which is precisely the region a
cost-quality paper is most likely to quote.

**Naive accounting is biased toward the router, systematically.** The naive error
is negative at every non-degenerate point: substituting per-model means always
*understated* true cost here, by 0.4% to 6.3%, and the bias grows as routing gets
more aggressive. The mechanism is the one Stage 8 predicts — the router sends
short, easy questions to the cheap model and the residual traffic on the
expensive model is longer than that model's average item, so the mean-cost
substitution charges too little for it. Any savings figure computed this way is
optimistic.

**The magnitude is much smaller than in our pipeline, and that is the
interesting part.** Our correction was 1.9x the size of true regret; here it is
at most ~6% of cost. The reason is visible in the token statistics: GSM8K answers
from both models are short and similar in length (Mixtral mean 132 output tokens,
p10–p90 68–213; GPT-4 mean 124, p10–p90 63–201), so within-model cost CV is only
~0.36–0.43. Our Tier 1 was a reasoning model emitting a mean of ~5,300 completion
tokens with an order-of-magnitude spread. **The size of the correction scales with
within-model cost dispersion**, which means the problem is getting worse, not
better, as the field routes among reasoning models with variable-length thinking
budgets. RouteLLM (2024-era, short-answer models) is close to the benign end of
this spectrum; a 2026 router over reasoning models is at the other end.

## FrugalGPT

Checked as the pre-registered fallback. Chen, Zaharia & Zou, *FrugalGPT: How to
Use Large Language Models While Reducing Cost and Improving Performance*
(arXiv:2305.05176, 2023) has no official code or data release from the authors;
there is no `stanford-futuredata/FrugalGPT` artifact containing per-query cost
traces. The paper reports aggregate cost reductions and per-model published
prices. Because it is a **cascade**, its cost accounting is in one respect more
exposed to this issue than RouteLLM's: cascade cost is the sum of all
attempts up to acceptance, so per-item cost varies both with response length and
with cascade depth. We could not verify this, and we are not going to estimate
it from the paper's aggregates.

## Scope and honest limitations of this stage

- **One benchmark, one router, one model pair.** GSM8K only, because it is the
  only RouteLLM artifact with response text. The BERT router is one of their five;
  the `mf` and `sw_ranking` routers require per-prompt OpenAI embedding calls and
  were not run.
- **Token counts are reconstructed, not published.** Tokenizing released
  response text recovers completion tokens accurately but cannot recover
  provider-side details such as system-prompt overhead, chat-template tokens, or
  billing rounding. Absolute costs are therefore approximate; the naive-vs-true
  *gap* is not sensitive to those details because both estimators use the same
  token counts.
- **Prices are ours.** The correction term's magnitude in dollars depends on the
  price ratio we assumed. The sign flip at 30% strong does not: it is driven by
  the covariance between routing and length, and survives any positive price
  vector with the strong model dearer than the weak.
- **This is not a criticism of RouteLLM's conclusions.** Their reported
  quality-vs-call-fraction curves are unaffected by any of this; only the
  translation of call fraction into cost is. That translation is nevertheless how
  the result is quoted, including in their own README.

## What would fix it, at negligible cost to publishers

Report per-item prompt and completion token counts alongside routing decisions.
Every provider returns them in the `usage` field of the response that the
evaluation already makes; storing two extra integers per row would have made this
entire stage a direct check rather than a partial reconstruction, and would let
any reader compute `R_true` instead of `R_naive`.
