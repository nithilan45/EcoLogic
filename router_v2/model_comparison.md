# R1 vs R2 ablation (Stage 2)

Both variants predict `P(tier t answers correctly | query)` for Tier 1 and Tier 2 with a per-tier logistic regression, and both compute their features entirely on CPU with **zero marginal API cost**, preserving EcoLogic's constraint that routing must not itself call an LLM.

This adapts the core idea of **RouteLLM** (Ong et al., *RouteLLM: Learning to Route LLMs from Preference Data*, ICLR 2025) -- a predicted success probability driving a cost threshold -- to a local feature space. It is **not** a reproduction of their method: RouteLLM learns from human preference data via a hosted preference model, whereas these variants fit directly on observed per-tier correctness using locally-computed features.

- **R1**: TF-IDF word 1-2 grams + char_wb n-grams + 25 hand-engineered features (length, code-syntax markers, question words, numeric content, multiple-choice markers).
- **R2**: `all-MiniLM-L6-v2` sentence embeddings (384-d, CPU) + the same 25 hand features.

Architecture and hyperparameters were chosen by 5-fold cross-validation **within TRAIN only** (n=840); CALIBRATION (n=360) was scored once, afterwards.

## Selection metric: mean ROC-AUC across the two tier heads (TRAIN CV)

| Variant | Best config | Tier 1 AUC | Tier 2 AUC | Mean AUC |
|---|---|---|---|---|
| R1 | char(2, 4), min_df=2, C=0.1 | 0.7323 | 0.5778 | **0.6550** |
| R2 | C=10.0 | 0.7415 | 0.6076 | **0.6746** |

**TRAIN-CV winner: R2.** This is the variant carried to Stage 3 and, once, to the frozen test set.

## Confirmation on CALIBRATION (scored once, both variants)

| Variant | Head | Base rate | AUC | Brier | Log loss |
|---|---|---|---|---|---|
| R1 | Tier 1 P(correct) | 0.856 | 0.7148 | 0.2062 | 0.5906 |
| R1 | Tier 2 P(correct) | 0.894 | 0.6907 | 0.1978 | 0.5878 |
| R2 | Tier 1 P(correct) | 0.856 | 0.6558 | 0.2061 | 0.6635 |
| R2 | Tier 2 P(correct) | 0.894 | 0.7010 | 0.1517 | 0.5066 |

An AUC near 0.5 means the head cannot tell which queries that tier will get right. Read these numbers before reading any downstream routing result: the whole learned-router approach depends on these heads carrying signal.

## Full TRAIN-CV grids

| Variant | char n-grams | min_df | C | Tier 1 AUC | Tier 2 AUC | Mean AUC |
|---|---|---|---|---|---|---|
| R1 | (2, 4) | 1 | 0.1 | 0.7312 | 0.5768 | 0.6540 |
| R1 | (2, 4) | 1 | 1.0 | 0.7378 | 0.5722 | 0.6550 |
| R1 | (2, 4) | 1 | 10.0 | 0.7354 | 0.5643 | 0.6499 |
| R1 | (2, 4) | 2 | 0.1 | 0.7323 | 0.5778 | 0.6550 |
| R1 | (2, 4) | 2 | 1.0 | 0.7389 | 0.5709 | 0.6549 |
| R1 | (2, 4) | 2 | 10.0 | 0.7299 | 0.5517 | 0.6408 |
| R1 | (3, 5) | 1 | 0.1 | 0.7309 | 0.5760 | 0.6535 |
| R1 | (3, 5) | 1 | 1.0 | 0.7376 | 0.5708 | 0.6542 |
| R1 | (3, 5) | 1 | 10.0 | 0.7354 | 0.5608 | 0.6481 |
| R1 | (3, 5) | 2 | 0.1 | 0.7324 | 0.5770 | 0.6547 |
| R1 | (3, 5) | 2 | 1.0 | 0.7389 | 0.5697 | 0.6543 |
| R1 | (3, 5) | 2 | 10.0 | 0.7305 | 0.5510 | 0.6407 |
| R2 | (embeddings) | - | 0.1 | 0.7298 | 0.5729 | 0.6513 |
| R2 | (embeddings) | - | 1.0 | 0.7392 | 0.5826 | 0.6609 |
| R2 | (embeddings) | - | 10.0 | 0.7415 | 0.6076 | 0.6746 |
