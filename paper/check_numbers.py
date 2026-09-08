#!/usr/bin/env python3
"""Verify that every headline number in the paper matches its artifact.

The provenance table in `README.md` says which committed JSON each claim comes
from. This script makes that table *executable*, in both directions.

**Forward.** For each curated claim, recompute the value from the artifact,
format it the way the paper does, and require that string to appear in
`main.tex` or `appendix.tex`. This catches a claim that was reworded until its
number silently vanished.

The forward direction is the guard that bites: it was written after finding a
`+1.15\\pp` learning-curve gain in four files that the artifact reports as
`+1.47\\pp`, and it catches exactly that --- re-run an experiment without
updating the prose and the old string stops matching.

**Coverage.** The forward direction says nothing about numbers it was never told
about, so the script also reports what fraction of the paper's numeric literals
are under audit at all, and prints the ones that are not. That list is the
honest measure of how far this check reaches, and shrinking it is how the check
gets stronger.

An earlier version of this script tried the reverse direction as a *check* ---
flagging any literal not derivable from some artifact value. That was dropped
because it cannot fail usefully: the artifacts contain ~1{,}500 distinct floats,
which after percent and sign variants generate ~15{,}000 candidate strings, more
than the ~9{,}900 two-decimal values in $[0,100]$. Every two-decimal number in
the paper matched something, including the stale `1.15`. A check that passes on
a known error is worse than no check, so it is a report now, not a check.

Neither direction can catch a correct number attributed to the wrong artifact,
or prose that misdescribes a correct number. This is a guard against drift, not
a substitute for reading the paper.

    python3 paper/check_numbers.py          # exits 1 on any failure
    python3 paper/check_numbers.py -v       # also print the passing checks
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


RB0 = load("stage11_13/s11_routerbench_0shot.json")
RB5 = load("stage11_13/s11_routerbench_5shot.json")
CEIL = load("stage11_13/s12_ceiling.json")
LC = load("stage11_13/s12b_learning_curve_0shot.json")
LLM = load("stage11_13/s13_llm_router.json")
ENC = load("stage11_13/s13b_encoder_finetune.json")
VAL = load("stage11_13/s11_validate.json")

# Also loaded purely to feed the reverse check, since the paper quotes earlier
# stages of the project alongside the new work.
EARLIER = [load(p) for p in (
    "stage7_10/s7_ceiling.json",
    "stage7_10/s9_static_baselines.json",
    "stage7_10/regret_correction_validation.json",
    "stage7_10/external_generalization.json",
    "stage11_13/s13c_encoder_routerbench_0shot.json",
)]

TEX = ""
for name in ("main.tex", "appendix.tex"):
    with open(os.path.join(HERE, name)) as f:
        TEX += f.read()

# LaTeX writes thousands separators as 36{,}494; normalise so a check can be
# written with the plain integer.
TEX_FLAT = TEX.replace("{,}", ",").replace("{,}", ",")


def tier(set_name, t, key):
    return CEIL["sets"][set_name]["per_tier"][t][key]


def reliabilities():
    return [tier(s, t, "reliability")
            for s in CEIL["sets"] for t in CEIL["sets"][s]["per_tier"]]


def auc_stars():
    return [tier(s, t, "auc_star_beta_moment_match")
            for s in CEIL["sets"] for t in CEIL["sets"][s]["per_tier"]]


def enc_sel(backbone, key):
    return ENC["results"][f"sentence-transformers/{backbone}"]["selection"][key]


def rung(name, key, beta="0.5"):
    r = LLM["rungs"][name]
    return r["betas"][beta][key] if key in r["betas"][beta] else r[key]


# Each check: (label, value recomputed from the artifact, format spec, artifact).
# These are *curated*: each one is a number the paper is known to state, so a
# missing one is a failure. The generated checks appended further down are not
# curated -- the paper is not obliged to quote every value in every artifact --
# so those only report NOT-QUOTED and raise coverage.
CURATED = [
    # --- Stage 11, RouterBench 0-shot (primary) ---
    ("kappa median (pp)", RB0 and RB0["H1_kappa_pp"]["median"], ".2f",
     "s11_routerbench_0shot.json"),
    ("kappa CI low", RB0 and RB0["H1_kappa_pp"]["lo"], ".2f", "same"),
    ("kappa CI high", RB0 and RB0["H1_kappa_pp"]["hi"], ".2f", "same"),
    ("best-router gain (pp)", RB0 and RB0["H2_best_router_gain_pp"]["median"], ".3f", "same"),
    ("best-router rho (%)", RB0 and 100 * RB0["H2_best_router_rho"]["median"], ".1f", "same"),
    ("best-router AUC", RB0 and RB0["out_of_fold_auc"]["tfidf_logreg"]["mean_auc"], ".4f", "same"),
    ("MiniLM+logreg AUC", RB0 and RB0["out_of_fold_auc"]["minilm_logreg"]["mean_auc"], ".4f", "same"),
    ("MiniLM+MLP AUC", RB0 and RB0["out_of_fold_auc"]["minilm_mlp"]["mean_auc"], ".4f", "same"),
    ("MiniLM+GBM AUC", RB0 and RB0["out_of_fold_auc"]["minilm_gbm"]["mean_auc"], ".4f", "same"),
    ("items analysed", RB0 and RB0["n_items"], ",d", "same"),
    ("models", RB0 and RB0["n_models"], "d", "same"),
    ("pairs with positive effect", RB0 and RB0["pairwise_tests"]["n_positive_effect"], "d", "same"),
    ("pairs raw p<.05", RB0 and RB0["pairwise_tests"]["n_raw_p_below_.05"], "d", "same"),
    ("pairs surviving BH", RB0 and RB0["pairwise_tests"]["n_significant_bh"], "d", "same"),
    ("pairs surviving Holm", RB0 and RB0["pairwise_tests"]["n_significant_holm"], "d", "same"),
    ("MMLU family n", RB0 and RB0["family_sizes"].get("mmlu"), ",d", "same"),
    ("GSM8K family n", RB0 and RB0["family_sizes"].get("gsm8k"), ",d", "same"),
    ("MBPP family n", RB0 and RB0["family_sizes"].get("mbpp"), ",d", "same"),

    # --- Stage 11, 5-shot replication ---
    ("5-shot kappa (pp)", RB5 and RB5["H1_kappa_pp"]["median"], ".2f",
     "s11_routerbench_5shot.json"),
    ("5-shot gain (pp)", RB5 and RB5["H2_best_router_gain_pp"]["median"], ".3f", "same"),
    ("5-shot rho (%)", RB5 and 100 * RB5["H2_best_router_rho"]["median"], ".1f", "same"),
    ("5-shot items analysed", RB5 and RB5["n_items"], ",d", "same"),
    ("5-shot items dropped", RB5 and RB5.get("n_items_dropped_missing_cells"), "d", "same"),
    ("5-shot pairs positive", RB5 and RB5["pairwise_tests"]["n_positive_effect"], "d", "same"),
    ("5-shot pairs Holm", RB5 and RB5["pairwise_tests"]["n_significant_holm"], "d", "same"),

    # --- Stage 12, replicates ---
    ("min reliability", CEIL and min(reliabilities()), ".3f", "s12_ceiling.json"),
    ("max reliability", CEIL and max(reliabilities()), ".3f", "same"),
    ("min AUC*", CEIL and min(auc_stars()), ".3f", "same"),
    ("Tier-1 pool reliability", CEIL and tier("pool_temp0.7", "tier1", "reliability"), ".3f", "same"),
    ("Tier-3 test reliability", CEIL and tier("test_temp0.0", "tier3", "reliability"), ".3f", "same"),
    ("ceiling / kappa ratio", CEIL and CEIL["H3"]["median_ceiling_over_kappa"], ".2f", "same"),

    # --- Stage 12b, learning curve (exploratory) ---
    ("learning-curve asymptote", LC and LC["fits"]["minilm_logreg"]["asymptote_A"], ".3f",
     "s12b_learning_curve_0shot.json"),
    ("AUC at 1e6", LC and LC["fits"]["minilm_logreg"]["auc_at_1e6"], ".3f", "same"),
    ("AUC at 1e9", LC and LC["fits"]["minilm_logreg"]["auc_at_1e9"], ".3f", "same"),
    ("gain at 8k (pp)", LC and LC["fits"]["minilm_logreg"]["gain_pp"][5], ".2f", "same"),
    ("gain at 29k (pp)", LC and LC["fits"]["minilm_logreg"]["gain_pp"][7], ".2f", "same"),
    ("AUC at 29k", LC and LC["fits"]["minilm_logreg"]["auc"][7], ".3f", "same"),

    # --- Stage 13, the ladder ---
    ("frozen reference AUC", LLM and LLM["baseline_stage7c"]["refit_here_mean_auc"], ".4f",
     "s13_llm_router.json"),
    ("frozen reference gain (pp)", LLM and LLM["baseline_stage7c"]["refit_here_gain_pp"], ".3f", "same"),
    ("Stage 7c best AUC", LLM and LLM["baseline_stage7c"]["s7c_best_mean_auc"], ".4f", "same"),
    ("prompted 0-shot AUC", LLM and rung("prompted_0shot", "mean_auc"), ".4f", "same"),
    ("prompted 4-shot AUC", LLM and rung("prompted_4shot", "mean_auc"), ".4f", "same"),
    ("prompted 0-shot gain (pp)", LLM and rung("prompted_0shot", "gain_pp"), ".3f", "same"),
    ("prompted 4-shot gain (pp)", LLM and rung("prompted_4shot", "gain_pp"), ".3f", "same"),
    ("split kappa (pp)", LLM and 100 * rung("prompted_0shot", "kappa"), ".2f", "same"),
    ("encoder L6 AUC", LLM and rung("finetuned_encoder_L6", "mean_auc"), ".4f", "same"),
    ("total Stage 11-13 spend", LLM and LLM["spend"]["usd"], ".2f", "same"),
    ("fine-tune training cost", LLM and LLM["spend"]["by_stage"]["finetune_training"]["usd"], ".2f", "same"),
    ("prompted 4-shot cost", LLM and LLM["spend"]["by_stage"]["prompted_4shot"]["usd"], ".2f", "same"),
    ("fine-tune tokens", LLM and LLM["state_ft1_gemma27b"]["token_count"], ",d", "same"),

    # --- Stage 13b, end-to-end encoders (exploratory) ---
    ("encoder L6 inner-val AUC", ENC and enc_sel("all-MiniLM-L6-v2", "val_auc"), ".3f",
     "s13b_encoder_finetune.json"),
    ("encoder L6 held-out AUC", ENC and enc_sel("all-MiniLM-L6-v2", "cal_auc"), ".4f", "same"),
    ("encoder L12 held-out AUC", ENC and enc_sel("all-MiniLM-L12-v2", "cal_auc"), ".4f", "same"),

    # --- Appendix B, validation ---
    ("proposition checks run", VAL and VAL["n_checks"], "d", "s11_validate.json"),
    ("proposition checks failed", VAL and VAL["n_checks"] - VAL["n_failures"], "d", "same"),
    ("naive variance inflation", VAL and VAL["checks"]
     ["P4_naive_estimator_is_inflated"]["mean_inflation_factor_of_naive_var"], ".2f", "same"),

    # --- Stage 13, C1/C2 verdicts ---
    ("C1 best delta vs Stage 7c", LLM and LLM["C1"]["delta_vs_s7c"], ".3f",
     "s13_llm_router.json"),
    ("C2 best new-rung gain (pp)", LLM and LLM["C2"]["best_gain_pp"], ".3f", "same"),
    ("C2 shortfall vs reference (pp)", LLM and LLM["C2"]["delta_pp"], ".3f", "same"),
]

# Generated: the per-router rows of Table 1 and their bootstrap bounds, the
# paired-bootstrap rows of Table 3, and every replicate cell. Written as loops
# because writing 60 near-identical tuples by hand is how a table and its
# checker drift apart.
CHECKS = []

if RB0:
    for rname in RB0["per_router_median"]:
        m = RB0["per_router_median"][rname]
        CHECKS += [
            (f"{rname} gain median", m["gain_pp"]["median"], ".3f", "s11_routerbench_0shot.json"),
            (f"{rname} gain lo", m["gain_pp"]["lo"], ".3f", "same"),
            (f"{rname} gain hi", m["gain_pp"]["hi"], ".3f", "same"),
            (f"{rname} rho median", m["rho"]["median"], ".3f", "same"),
            (f"{rname} cells positive", m["n_pairs_positive_gain"], "d", "same"),
        ]

if RB5:
    CHECKS += [
        ("5-shot best AUC", RB5["out_of_fold_auc"]["tfidf_logreg"]["mean_auc"], ".4f",
         "s11_routerbench_5shot.json"),
        ("5-shot kappa lo", RB5["H1_kappa_pp"]["lo"], ".2f", "same"),
        ("5-shot kappa hi", RB5["H1_kappa_pp"]["hi"], ".2f", "same"),
        ("5-shot gain lo", RB5["H2_best_router_gain_pp"]["lo"], ".3f", "same"),
        ("5-shot gain hi", RB5["H2_best_router_gain_pp"]["hi"], ".3f", "same"),
        ("5-shot rho lo (%)", 100 * RB5["H2_best_router_rho"]["lo"], ".1f", "same"),
        ("5-shot rho hi (%)", 100 * RB5["H2_best_router_rho"]["hi"], ".1f", "same"),
        ("5-shot pairs BH", RB5["pairwise_tests"]["n_significant_bh"], "d", "same"),
    ]

if LLM and "paired_bootstrap_vs_frozen" in LLM:
    for rname, b in LLM["paired_bootstrap_vs_frozen"].items():
        CHECKS += [
            (f"{rname} dAUC", b["delta_mean_auc"], ".3f", "s13_llm_router.json"),
            (f"{rname} dAUC lo", b["lo"], ".3f", "same"),
            (f"{rname} dAUC hi", b["hi"], ".3f", "same"),
            (f"{rname} Holm p", b["p_holm"], ".3f", "same"),
        ]

if CEIL:
    for sname, s in CEIL["sets"].items():
        for t, v in s["per_tier"].items():
            CHECKS += [
                (f"{sname} {t} reliability", v["reliability"], ".3f", "s12_ceiling.json"),
                (f"{sname} {t} AUC*", v["auc_star_beta_moment_match"], ".3f", "same"),
            ]


def fmt(v, spec):
    if v is None:
        return None
    if spec.endswith("d"):
        return format(int(round(v)), spec)
    return format(float(v), spec)


def renderings(v, spec):
    """Every rounding of `v` the paper could legitimately use.

    A Holm-adjusted p of 0.256 may be quoted as 0.256 or 0.26; both are correct
    roundings of the same value, and neither admits a different value, so
    accepting any of them does not weaken the check. What it must not accept is
    a *mis*-rounding, and it cannot: each rounding is a function of the value.

    The relaxation stops at two decimals and at one step. Going further
    degenerates: 0.0289 rounds to "0.0" at one decimal, and "0.0" occurs in
    every table in the paper, so a one-decimal fallback would make the check
    pass on anything.
    """
    if v is None:
        return []
    if spec.endswith("d"):
        s = fmt(v, spec)
        return [s, s.replace(",", "{,}")]
    d = int(spec[1])
    return [format(float(v), f".{k}f") for k in {d, max(d - 1, 2)}]


def occurs(s):
    """Is `s` in the paper as a whole number, not as part of a longer one?

    Plain substring matching is useless here: `0.71` occurs inside `0.7160`, so
    a perturbed AUC of `0.7205` would "match" via its two-decimal rendering
    `0.72` if that string happened to sit inside some other number.
    """
    pat = r"(?<![\d.])" + re.escape(s) + r"(?![\d])"
    return re.search(pat, TEX_FLAT) is not None


# ------------------------------------------------------------- coverage report

def coverage(audited):
    """Which numeric literals in the paper are covered by a curated check.

    Restricted to literals with two or more decimals: integers and one-decimal
    numbers in this paper are overwhelmingly structural (section numbers, "3
    epochs", "5,000 prompts", beta grids), so listing them would bury the
    empirical ones we actually want eyes on.
    """
    lits = re.findall(r"(?<![\d.])\d{1,3}(?:,\d{3})*\.\d{2,4}(?![\d])", TEX_FLAT)
    uncovered = {}
    for lit in lits:
        plain = lit.replace(",", "")
        if plain in audited:
            continue
        # A curated check at 4 decimals also covers the same number quoted at 2
        # or 3 in prose.
        if any(a.startswith(plain) or plain.startswith(a) for a in audited):
            continue
        uncovered[plain] = uncovered.get(plain, 0) + 1
    return uncovered, len(lits)


def main():
    verbose = "-v" in sys.argv
    rows, bad, skipped, unquoted, audited = [], 0, 0, 0, set()
    for required, (label, value, spec, src) in (
            [(True, c) for c in CURATED] + [(False, c) for c in CHECKS]):
        s = fmt(value, spec)
        if s is None:
            rows.append(("SKIP", label, "-", src))
            skipped += 1
            continue
        present = any(occurs(c) for c in renderings(value, spec))
        if present:
            rows.append(("OK", label, s, src))
            audited.add(s.replace(",", ""))
        elif required:
            rows.append(("ABSENT", label, s, src))
            bad += 1
        else:
            rows.append(("UNQUOTED", label, s, src))
            unquoted += 1

    w = max(len(r[1]) for r in rows)
    for status, label, s, src in rows:
        if status == "OK" and not verbose:
            continue
        if status == "UNQUOTED" and not verbose:
            continue
        print(f"[{status:8}] {label:<{w}}  {s:>10}  ({src})")

    n_ok = sum(1 for r in rows if r[0] == "OK")
    print(f"curated: {len(CURATED) - bad - skipped}/{len(CURATED)} values the paper is "
          f"known to state were found"
          f"{f', {skipped} skipped (artifact absent or key renamed)' if skipped else ''}"
          f"{f', {bad} ABSENT' if bad else ''}")
    print(f"generated: {n_ok - (len(CURATED) - bad - skipped)}/{len(CHECKS)} further "
          f"artifact values also appear ({unquoted} are simply not quoted, which is "
          f"not an error)")

    uncovered, n_lits = coverage(audited)
    n_cov = n_lits - sum(uncovered.values())
    print(f"coverage: {n_cov}/{n_lits} multi-decimal literals in the paper are under "
          f"explicit audit ({100 * n_cov / max(n_lits, 1):.0f}%). "
          f"{len(uncovered)} distinct values are not:")
    for lit, n in sorted(uncovered.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {lit:>10}  x{n}")
    print("  Uncovered is not wrong -- most are interval bounds, per-model AUCs and\n"
          "  earlier-stage figures. It is the list to extend CHECKS with.")

    if bad:
        print("\nABSENT means the artifact's value does not appear in main.tex or "
              "appendix.tex. Either the paper is stale or the number moved; fix "
              "one of the two.")
        raise SystemExit(1)
    print("\nEvery audited number in the paper matches its artifact.")


if __name__ == "__main__":
    main()
