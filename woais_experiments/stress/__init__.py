"""Systems stress tests for EcoLogic routing under load.

MEASURED (real endpoint) and SIMULATED (DES) runs are never mixed.
"""

from __future__ import annotations

MEASURED = "MEASURED"
SIMULATED = "SIMULATED"

POLICY_SPECS = (
    {"name": "always_cheap", "measured": "direct_cheap", "simulated": "always_cheap"},
    {"name": "always_strong", "measured": "direct_strong", "simulated": "always_strong"},
    {"name": "ecologic", "measured": "ecologic", "simulated": "ecologic"},
    {"name": "cost_matched_static", "measured": "static_mixture", "simulated": "cost_matched_static"},
)

RESULTS_MEASURED = "stress/measured"
RESULTS_SIMULATED = "stress/simulated"

__all__ = [
    "MEASURED",
    "SIMULATED",
    "POLICY_SPECS",
    "RESULTS_MEASURED",
    "RESULTS_SIMULATED",
]
