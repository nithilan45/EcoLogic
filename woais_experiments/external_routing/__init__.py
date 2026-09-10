"""External-router validation: real assignments, naive vs realized cost.

The evaluated policy is an independently produced router (default: RouteLLM
BERT). Oracle assignments are computed only as a labeled upper bound and never
choose the evaluated route.
"""

from woais_experiments.external_routing.reconstruct_assignments import (
    EXTERNAL_LEARNED_ROUTER,
    ORACLE_ASSIGNMENT,
    oracle_route,
    route_by_score,
)

__all__ = [
    "EXTERNAL_LEARNED_ROUTER",
    "ORACLE_ASSIGNMENT",
    "oracle_route",
    "route_by_score",
]
