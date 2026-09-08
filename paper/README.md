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

As of the current draft it compiles to **12 pages**: body through §Conclusion
ending a third of the way down page 7, references filling the rest of page 7,
and the appendix on pages 8–12. So the body is **≈6.3 pages** under
`article` + 1 inch margins.

Three trims have already been taken to get there, and each moved material to the
appendix rather than deleting it: the matched-cost/cost-axis section is now
Appendix H (it is prior work of this project, not a contribution of this paper),
Proposition 4's estimator now states its conclusion in the body with the formula
and proof in Appendix A, and the replicate table is now Appendix G — where it
gained the two Tier-3 `AUC*` cells the body version omitted.

**Expect the count to drop under the venue style.** `article` with 1 inch margins
is looser than most conference styles; ICLR's own style file is denser, so the
same body is likely to land near 5 pages. Re-measure before cutting further.

If it does not, cut in this order:

1. Table 3's realised-gain column moves to Appendix E, keeping only ΔAUC in the
   body. The AUC-vs-gain dissociation would then need one sentence with the two
   numbers inline.
2. Related work compresses to four sentences; the survey citations
   (`yuan2025whoroutes`, `huang2025routereval`) stay, since the paper positions
   itself against them.
3. §4's two "things that do survive" collapse to one sentence each.

Do **not** cut the deviations paragraph in §5, the blocked-rung paragraph, or
the "null results are not proofs of absence" limitation. Those are the parts a
reviewer is entitled to see.

## Where every number comes from

No number in the paper is typed by hand from memory; each traces to a committed
artifact.

| Paper location | Claim | Artifact |
|---|---|---|
| §3, Table 1 | median `kappa` = 12.11 pp; router gain 0.59 pp; `rho_realised` = 4.6%; out-of-fold AUCs | `stage11_13/s11_routerbench_0shot.json` |
| §3, multiple comparisons | 49/55 positive, 28 raw, 26 BH, 13 Holm | same, `pairwise_tests` |
| §3, per-pair detail | 440 cells × 9 operating points | `stage11_13/s11_routerbench_pairs_0shot.csv.gz` |
| §3, 5-shot replication; Appendix D table | `kappa` 11.44 pp, gain 0.640 pp, `rho` 5.5%, 50/55 positive, 22 BH, 11 Holm, 28 items dropped | `stage11_13/s11_routerbench_5shot.json` |
| §4, Table 2 | reliability 0.77–0.98; `AUC*` 0.95–0.99 | `stage11_13/s12_ceiling.json` |
| §4 | ceiling 8.4–15.6 pp vs `kappa` 3.3–9.5 pp, ratio 2.82 | same, `H3` |
| §4 | peeking policy captures 34–55% of `kappa` | same, `split_replicate_gain` |
| §5 | learning-curve asymptote 0.741, AUC(1e6)=0.726 | `stage11_13/s12b_learning_curve_0shot.json` |
| §5, Table 3 | classical rungs (k-NN … logistic, Stage 7c) | `stage7_10/s7_ceiling.json` |
| §5, Table 3 | frozen reference refit 0.6896; prompted 70B 0.6416 / 0.7105; gains at β=0.5; C1 and C2 verdicts | `stage11_13/s13_llm_router.json` |
| §5, Table 3 | ΔAUC intervals and Holm-adjusted p-values | same, `paired_bootstrap_vs_frozen` |
| §5 | end-to-end encoders 0.6945 / 0.6912, inner-val 0.751 | `stage11_13/s13b_encoder_finetune.json` |
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
- The fine-tuned **generative** LLM rung is reported as **BLOCKED** with the
  provider's four verbatim errors (Appendix F), not estimated. The `\FTAUC` /
  `\FTINFER` / `\FTTOTAL` macros in `main.tex` carry that state explicitly in
  the source. The $25 spend gate was never the binding constraint; the
  provider's account balance was, at $9.71.
- The paper reports that **no** router rung's AUC advantage over the frozen
  baseline survives Holm correction, and separately that the widest interval is
  ±0.05, so a +0.02 effect cannot be ruled out. Both statements are in §5 and
  Limitations.
