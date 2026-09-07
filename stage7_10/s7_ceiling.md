# Stage 7c — is the predictability ceiling structural or just linear?

The project's central negative claim is that per-item tier success is only weakly predictable from the prompt. That claim rested on two linear models, so the obvious objection is that we tested a weak router and blamed the workload. Here the same representation, labels and splits are given to stronger learners.

TRAIN n = 3500, CALIBRATION n = 1500. Fit on TRAIN, scored once on CALIBRATION. The frozen test set is untouched.

| Model | Tier 1 AUC | Tier 2 AUC | Mean CALIB AUC | Mean TRAIN AUC |
|---|---|---|---|---|
| logistic (R2 baseline) | 0.6711 | 0.6922 | **0.6816** | 0.8140 |
| gradient boosting | 0.6286 | 0.6808 | **0.6547** | 0.9577 |
| random forest | 0.6523 | 0.6882 | **0.6703** | 0.9999 |
| k-NN (k=50) | 0.6136 | 0.6906 | **0.6521** | 0.7835 |

Best non-linear model: **random forest**, mean CALIBRATION AUC 0.6703 versus the linear head's 0.6816 — a lift of **-0.0114**.

**Verdict: ceiling is structural: no stronger learner beats the linear head by >= 0.02 AUC on held-out data.**

The TRAIN column is the useful diagnostic. Where a model's train AUC runs far above its calibration AUC it has capacity to spare and is spending it on memorisation, which means the held-out ceiling is a property of the signal available in the prompt rather than of model capacity.
