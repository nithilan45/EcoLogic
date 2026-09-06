# Stage 7 — R1 vs R2 ablation on the scaled, majority-voted pool

Same two variants and the same selection procedure as Stage 2, refit on **3500 TRAIN items** (Stage 2: 840) with per-tier labels from a **majority of k=3 samples at temperature 0.7** instead of a single temperature-0 sample. Selection is 5-fold cross-validation within TRAIN only; CALIBRATION (n=1500) is scored once, afterwards, for both variants.

## TRAIN-CV selection metric: mean ROC-AUC across the two tier heads

| Variant | Best config | Tier 1 AUC | Tier 2 AUC | Mean AUC | Stage 2 mean AUC |
|---|---|---|---|---|---|
| R1 | char(2, 4), min_df=2, C=0.1 | 0.7262 | 0.7047 | **0.7154** | 0.6550 |
| R2 | C=0.1 | 0.7275 | 0.7048 | **0.7162** | 0.6746 |

**TRAIN-CV winner: R2.**

The Stage 2 column is the direct test of the data-scaling hypothesis at the level of the probability heads: if the calibration gap were caused by insufficient or noisy labels, these AUCs should be visibly higher than Stage 2's.

## Confirmation on CALIBRATION (scored once, both variants)

| Variant | Head | Base rate | AUC | Brier | Log loss |
|---|---|---|---|---|---|
| R1 | Tier 1 P(correct) | 0.921 | 0.6520 | 0.1861 | 0.5524 |
| R1 | Tier 2 P(correct) | 0.901 | 0.7004 | 0.1874 | 0.5564 |
| R2 | Tier 1 P(correct) | 0.921 | 0.6650 | 0.2031 | 0.5917 |
| R2 | Tier 2 P(correct) | 0.901 | 0.7015 | 0.2020 | 0.5896 |

## Full TRAIN-CV grids

| Variant | char n-grams | min_df | C | Tier 1 AUC | Tier 2 AUC | Mean AUC |
|---|---|---|---|---|---|---|
| R1 | (2, 4) | 1 | 0.1 | 0.7259 | 0.7039 | 0.7149 |
| R1 | (2, 4) | 1 | 1.0 | 0.7247 | 0.7053 | 0.7150 |
| R1 | (2, 4) | 1 | 10.0 | 0.7128 | 0.6913 | 0.7021 |
| R1 | (2, 4) | 2 | 0.1 | 0.7262 | 0.7047 | 0.7154 |
| R1 | (2, 4) | 2 | 1.0 | 0.7217 | 0.7028 | 0.7122 |
| R1 | (2, 4) | 2 | 10.0 | 0.7051 | 0.6831 | 0.6941 |
| R1 | (3, 5) | 1 | 0.1 | 0.7234 | 0.7036 | 0.7135 |
| R1 | (3, 5) | 1 | 1.0 | 0.7202 | 0.7063 | 0.7133 |
| R1 | (3, 5) | 1 | 10.0 | 0.7074 | 0.6938 | 0.7006 |
| R1 | (3, 5) | 2 | 0.1 | 0.7239 | 0.7045 | 0.7142 |
| R1 | (3, 5) | 2 | 1.0 | 0.7170 | 0.7030 | 0.7100 |
| R1 | (3, 5) | 2 | 10.0 | 0.6997 | 0.6843 | 0.6920 |
| R2 | (embeddings) | - | 0.1 | 0.7275 | 0.7048 | 0.7162 |
| R2 | (embeddings) | - | 1.0 | 0.6997 | 0.6858 | 0.6928 |
| R2 | (embeddings) | - | 10.0 | 0.6474 | 0.6511 | 0.6492 |
