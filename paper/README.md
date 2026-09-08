# Workshop paper draft

**Complementarity is Abundant, Predictability is Not the Problem: Decomposing
Why LLM Routers Gain So Little.** Target venue: an ICLR 2027 workshop.

## Build

```bash
sudo apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-fonts-recommended
cd paper && pdflatex main.tex && pdflatex main.tex   # twice, for cross-refs
```

`main.tex` is deliberately self-contained: `article` class plus `geometry`, so it
compiles in a minimal TeX install with no style files to fetch. **Before
submission, swap the preamble for the official workshop style file** (e.g.
`\usepackage{iclr2027_workshop}`) and re-check the page budget.

As of the current draft it compiles to **13 pages**: the body runs §1–§8 and ends
at the foot of page 7, references occupy page 8, and Appendices A–J fill pages
8–13. So the body is **exactly 7 pages** under `article` + 1 inch margins.

Six trims have been taken to get there, and each moved material to the appendix
rather than deleting it:

1. The matched-cost/cost-axis section is now Appendix J — it is prior work of
   this project, not a contribution of this paper.
2. Proposition 4's estimator states its conclusion in the body, with the formula
   and proof in Appendix A.
3. The replicate table is now Appendix H, where it gained the two Tier-3 `AUC*`
   cells the body version omitted.
4. The reliability and Bayes-AUC paragraphs of §4 were merged; they had been
   quoting the same two ranges twice.
5. Four limitations that state assumptions rather than caveats — replicate
   exchangeability, which propositions hold for general `K`, the status of the
   power-law fit, and the three-model scope — are now Appendix I. The four a
   reviewer most needs stay in the body.
6. The contributions list was compressed where it restated §3–§5 verbatim.

**Expect the count to drop under the venue style.** `article` with 1 inch margins
is looser than most conference styles; ICLR's own style file is denser, so the
same body should land nearer 5.5–6 pages. Re-measure before cutting further, and
note honestly that **this may still exceed a 4- or 5-page limit**, in which case
cut in this order:

1. Table 3's realised-gain column moves to Appendix E, keeping only ΔAUC in the
   body. The AUC-vs-gain dissociation would then need one sentence with the two
   numbers inline.
2. §4's two "things that do survive" (the peeking policy, the noise share)
   collapse to one sentence each with the detail in Appendix H.
3. §2 drops the statements of Propositions 3 and 5 to one line each, since
   Proposition 3's bound is reported as vacuous anyway.
4. Related work compresses to four sentences; the survey citations
   (`yuan2025whoroutes`, `huang2025routereval`) stay, since the paper positions
   itself against them.

Do **not** cut the deviations paragraph in §5, the blocked-rung paragraph, the
"null results are not proofs of absence" limitation, or the one-split caveat on
Table 2. Those are the parts a reviewer is entitled to see.

## Where every number comes from

No number in the paper is typed by hand from memory; each traces to a committed
artifact. That claim is **checked, not asserted**:

```bash
python3 paper/check_numbers.py      # exits 1 on any mismatch
python3 paper/check_numbers.py -v   # also lists the passing and unquoted checks
```

The script recomputes 73 curated values from the artifacts, formats them the way
the paper does, and requires each to appear in `main.tex` or `appendix.tex`;
56 further values are generated from the result JSONs (per-router table rows,
bootstrap bounds, paired-bootstrap rows, every replicate cell) and reported but
not required, since the paper is not obliged to quote everything. It also
reports **coverage**: what share of the paper's multi-decimal literals are under
explicit audit, currently **74%**, with the remainder listed so the gap is
visible rather than implied — mostly interval bounds, per-model AUCs and
earlier-stage figures.

Two things it deliberately does *not* claim. It cannot catch a correct number
attributed to the wrong artifact, or prose that misdescribes a correct number.
And an earlier version tried the reverse direction as a check — flag any literal
no artifact produces — which was removed because it could not fail usefully: the
artifacts hold ~1,500 distinct floats, generating more candidate strings than
there are two-decimal values in [0, 100], so it passed on a known error. That
error was real: a learning-curve gain of `+1.15 pp` appeared in four files where
the artifact says `+1.47 pp`. The curated direction catches it; the script is
regression-tested against that case and three perturbed AUCs.

| Paper location | Claim | Artifact |
|---|---|---|
| §3, Table 1 | median `kappa` = 12.11 pp; router gain 0.59 pp; `rho_realised` = 4.6%; out-of-fold AUCs | `stage11_13/s11_routerbench_0shot.json` |
| §3, multiple comparisons | 49/55 positive, 28 raw, 26 BH, 13 Holm | same, `pairwise_tests` |
| §3, per-pair detail | 440 cells × 9 operating points | `stage11_13/s11_routerbench_pairs_0shot.csv.gz` |
| §3, 5-shot replication; Appendix D table | `kappa` 11.44 pp, gain 0.640 pp, `rho` 5.5%, 50/55 positive, 22 BH, 11 Holm, 28 items dropped | `stage11_13/s11_routerbench_5shot.json` |
| §4, Appendix H table | reliability 0.77–0.98; `AUC*` 0.95–0.99 | `stage11_13/s12_ceiling.json` |
| §4 | ceiling 8.4–15.6 pp vs `kappa` 3.3–9.5 pp, ratio 2.82 | same, `H3` |
| §4 | peeking policy captures 34–55% of `kappa` | same, `split_replicate_gain` |
| §5 | learning-curve asymptote 0.741, AUC(1e6)=0.726, gain +1.47 pp at 8k → +2.10 pp at 29k | `stage11_13/s12b_learning_curve_0shot.json` |
| §5, Table 3 | classical rungs (k-NN … logistic, Stage 7c) | `stage7_10/s7_ceiling.json` |
| §5, Table 3 | frozen reference refit 0.6896; prompted 70B 0.6416 / 0.7105; gains at β=0.5; C1 and C2 verdicts | `stage11_13/s13_llm_router.json` |
| §5, Table 3 | ΔAUC intervals and Holm-adjusted p-values | same, `paired_bootstrap_vs_frozen` |
| §5 | end-to-end encoders 0.6945 / 0.6912, inner-val 0.751 | `stage11_13/s13b_encoder_finetune.json` |
| **§5, Table 2** | **the AUC/gain dissociation at scale:** unfrozen 0.7126 / +6.507 pp / 32.6% vs frozen logistic 0.7078 / +1.939 pp / 9.7% and frozen MLP 0.6617 / +2.825 pp / 14.2%; `kappa` = 19.95 pp; 27,735 / 1,460 / 7,299 items | `stage11_13/s13c_encoder_routerbench_0shot.json` |
| §5, Appendix F | the blocked fine-tuned generative rung, four provider errors | `stage11_13/s13_ftblocked.json`, `s13_endpoint_probe.json` |
| §6 | 86.8% vs 92.3% at 6.87× cost; 88.6% vs 55.0% | `raw_results/tables_cost.md` |
| §6 | RouteLLM +0.57 pp, 8/9 points, sign test p=0.039 | `stage7_10/s9_static_baselines.json` |
| §6 | correction 1.9× regret, reconciles to 2e-16, 6.3% on RouteLLM | `stage7_10/regret_correction_validation.json`, `external_generalization.json` |
| Appendix B | 14/14 proposition checks | `stage11_13/s11_validate.json` |
| Appendix C | pre-registration and 8 deviations | `stage11_13/prereg_stage11_13.md`, `DEVIATIONS.md` |
| Appendix E | cost ledger | `stage11_13/s13_spend.json` |

Theory and proofs: `stage11_13/theory.md` (the long form; the paper's §2 and
Appendix A are condensed from it).

Reproduction: `stage7_10/reproducibility_manifest.md` §12 gives every seed,
model string, hyperparameter, split and command for the work in this paper.
RouterBench itself is gitignored (~270 MB); `stage11_13/fetch_routerbench.py`
downloads both releases and **fails** unless their SHA-256 matches the files the
results were computed on.

## Honesty notes carried into the draft

- The pre-registered hypothesis H3 was **refuted** and the paper says so in the
  abstract, §4 and Appendix C. The ceiling of Proposition 3 is reported as
  vacuous at these effect sizes rather than quietly dropped.
- One pre-registration statement was **wrong** (D2: a peeking policy does not
  upper-bound `A†`). It is corrected in place and the affected quantity is
  relabelled.
- The learning-curve analysis and the end-to-end encoder rungs are
  **exploratory and not pre-registered**, and are labelled as such in the paper.
  The encoders are still judged against the pre-registered +0.05 threshold, so
  adding them cannot make the criterion easier to pass.
- **Table 2 qualifies the paper's own negative result, and the paper says so
  rather than burying it.** The encoder-at-scale rung triples matched-cost gain,
  which is evidence *against* a strong reading of "the gap does not close". It
  is reported in the abstract, §5, the conclusion and Limitations, with its
  caveats stated in the same breath: one split, one seed, three epochs, no
  interval, not pre-registered. The negative claim is correspondingly narrowed
  to AUC rather than left overstated.
- The fine-tuned **generative** LLM rung is reported as **BLOCKED** with the
  provider's four verbatim errors (Appendix F), not estimated. The `\FTAUC` /
  `\FTINFER` / `\FTTOTAL` macros in `main.tex` carry that state explicitly in
  the source. The $25 spend gate was never the binding constraint; the
  provider's account balance was, at $9.71.
- The paper reports that **no** router rung's AUC advantage over the frozen
  baseline survives Holm correction, and separately that the widest interval is
  ±0.05, so a +0.02 effect cannot be ruled out. Both statements are in §5 and
  Limitations.
