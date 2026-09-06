# Energy-regret verification (Stage 4)

Energy regret is the extra energy the learned router spends per item relative to the oracle tier choice, on the CALIBRATION split (n=360, variant R2, threshold tau=0.2245).

## Two independent computations

Both use per-tier mean energies as the constants `e_t`, which is what makes the confusion-matrix identity exact:

| Tier | mean energy per item (J) |
|---|---|
| 1 | 2.1424 |
| 2 | 1.0647 |
| 3 | 14.3293 |

| Method | Formula | Result (J/item) |
|---|---|---|
| A — direct, paired samples | `mean_i( e[that_i] - e[t*_i] )` | **-0.2496232176** |
| B — confusion matrix | `sum_{i!=j} P(t*=i, that=j) * (e_j - e_i)` | **-0.2496232176** |

**Absolute difference: 8.33e-17 — they match.** The two are algebraically the same quantity, so agreement to floating-point precision is a correctness check on the confusion matrix and the pairing, not independent evidence about the router.

Using each item's **actual** energy instead of per-tier means gives **0.2840 J/item**. This differs from the two figures above because energy is item-dependent (token counts vary per item), so the constant-`e_t` identity no longer holds. It is the more faithful number for real energy accounting; the constant version is the one the confusion-matrix formula applies to.

## Confusion matrix, P(oracle tier = i, router tier = j)

| oracle t* \ router t^ | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Tier 1 | 0.0389 | 0.2944 | 0.0139 |
| Tier 2 | 0.0083 | 0.6111 | 0.0111 |
| Tier 3 | 0.0000 | 0.0194 | 0.0028 |

Off-diagonal mass above the diagonal is false escalation (router picked a costlier tier than needed); below it is false de-escalation (router picked a cheaper tier than the oracle, which costs accuracy rather than energy).
