"""Held-out EcoLogic routing protocol.

The original Stage 1–2 n=364 panel is not this experiment. This package
splits the disjoint router_v2 graded pool (n=1200) 60/20/20, fits on train,
selects on validation, and scores the test set once.
"""

from woais_experiments.heldout.build_split import (
    HELDOUT_DIR,
    MANIFEST_PATH,
    SPLIT_SEED,
    load_manifest,
    load_panel,
)

__all__ = [
    "HELDOUT_DIR",
    "MANIFEST_PATH",
    "SPLIT_SEED",
    "load_manifest",
    "load_panel",
]
