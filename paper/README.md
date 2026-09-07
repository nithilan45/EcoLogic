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
`\usepackage{iclr2027_workshop}`) and re-check the page budget — the content is
sized for ~5 pages of body text under a typical workshop template, with
everything else in `appendix.tex`.

## Where every number comes from

No number in the paper is typed by hand from memory; each traces to a committed
artifact.

| Paper location | Claim | Artifact |
|---|---|---|
| §3, Table 1 | median `kappa` = 12.11 pp; router gain 0.59 pp; `rho_realised` = 4.6%; out-of-fold AUCs | `stage11_13/s11_routerbench_0shot.json` |
| §3, multiple comparisons | 49/55 positive, 28 raw, 26 BH, 13 Holm | same, `pairwise_tests` |
| §3, per-pair detail | 440 cells × 9 operating points | `stage11_13/s11_routerbench_pairs_0shot.csv.gz` |
| §4, Table 2 | reliability 0.77–0.98; `AUC*` 0.95–0.99 | `stage11_13/s12_ceiling.json` |
| §4 | ceiling 8.4–15.6 pp vs `kappa` 3.3–9.5 pp, ratio 2.82 | same, `H3` |
| §4 | peeking policy captures 34–55% of `kappa` | same, `split_replicate_gain` |
| §5 | learning-curve asymptote 0.741, AUC(1e6)=0.726 | `stage11_13/s12b_learning_curve_0shot.json` |
| §5, Table 3 | classical rungs (k-NN … logistic) | `stage7_10/s7_ceiling.json` |
| §5, Table 3 | prompted 70B zero-shot 0.6416, 4-shot 0.7105; fine-tune | `stage11_13/s13_llm_router.json` |
| §6 | 86.8% vs 92.3% at 6.87× cost; 88.6% vs 55.0% | `raw_results/tables_cost.md` |
| §6 | RouteLLM +0.57 pp, 8/9 points, sign test p=0.039 | `stage7_10/s9_static_baselines.json` |
| §6 | correction 1.9× regret, reconciles to 2e-16, 6.3% on RouteLLM | `stage7_10/regret_correction_validation.json`, `external_generalization.json` |
| Appendix B | 14/14 proposition checks | `stage11_13/s11_validate.json` |
| Appendix C | pre-registration and 5 deviations | `stage11_13/prereg_stage11_13.md`, `DEVIATIONS.md` |
| Appendix E | cost ledger | `stage11_13/s13_spend.json` |

Theory and proofs: `stage11_13/theory.md` (the long form; the paper's §2 and
Appendix A are condensed from it).

## Honesty notes carried into the draft

- The pre-registered hypothesis H3 was **refuted** and the paper says so in the
  abstract, §4 and Appendix C. The ceiling of Proposition 3 is reported as
  vacuous at these effect sizes rather than quietly dropped.
- One pre-registration statement was **wrong** (D2: a peeking policy does not
  upper-bound `A†`). It is corrected in place and the affected quantity is
  relabelled.
- The learning-curve analysis is **exploratory and not pre-registered**, and is
  labelled as such in the paper.
- If the Together AI fine-tune had failed or exhausted credits, the rung would
  be reported as **BLOCKED** with the provider error, not estimated. The
  `\FTAUC` / `\FTINFER` / `\FTTOTAL` macros in `main.tex` exist so that state is
  explicit in the source.
