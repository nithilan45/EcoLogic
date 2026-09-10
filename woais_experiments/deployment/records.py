"""Measured request logs for the deployment benchmark.

Every record is tagged ``measurement_type="MEASURED"``. Simulator fields are
refused at analysis time; this module never writes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

MEASUREMENT_TYPE = "MEASURED"

# JSON keys for the fields the benchmark is required to record.
REQUIRED_REQUEST_FIELDS = (
    "request_id",
    "timestamp",
    "query_length_chars",
    "input_tokens",
    "router_decision_ms",
    "provider_request_ms",
    "time_to_first_token_ms",
    "generation_ms",
    "end_to_end_ms",
    "selected_model",
    "output_tokens",
    "realized_provider_cost",
    "http_status",
    "error_type",
    "retry_count",
)

LIFECYCLE_FIELDS = (
    "lifecycle_state",
    "lifecycle_basis",
    "process_id",
    "request_index_in_process",
    "process_uptime_s",
)

SIMULATOR_KEYS = frozenset({
    "des_wait_s",
    "des_sojourn_s",
    "simulated_sojourn_s",
    "cold_penalty_s",
    "n_servers",
    "arrival_process",
    "idle_timeout_s",
    "simulator",
    "simulation_seed",
    "modeled_cold_start",
})

SIMULATOR_MEASUREMENT_TYPES = frozenset({
    "SIMULATED",
    "MODELED",
    "DERIVED",
    "IMPUTED",
})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class RequestRecord:
    """One measured inference attempt (client or in-process)."""

    request_id: str
    timestamp: str
    query_length_chars: int
    input_tokens: int | None
    router_decision_ms: float | None
    provider_request_ms: float | None
    time_to_first_token_ms: float | None
    generation_ms: float | None
    end_to_end_ms: float | None
    selected_model: str | None
    output_tokens: int | None
    realized_provider_cost: float | None
    http_status: int
    error_type: str | None
    retry_count: int
    measurement_type: str = MEASUREMENT_TYPE
    lifecycle_state: str | None = None
    lifecycle_basis: str | None = None
    process_id: int | None = None
    request_index_in_process: int | None = None
    process_uptime_s: float | None = None
    setup: str | None = None
    setup_name: str | None = None
    policy: str | None = None
    concurrency: int | None = None
    generation_mode: str | None = None
    backend: str | None = None
    environment_kind: str | None = None
    cloud_measured: bool | None = None
    cost_scope: str | None = None
    prompt_id: str | None = None
    selected_tier: int | None = None
    router_reason: str | None = None
    input_tokens_source: str | None = None
    output_tokens_source: str | None = None
    cost_basis: str | None = None
    price_slug: str | None = None
    price_source: str | None = None
    provider_name: str | None = None
    paid_api: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.measurement_type != MEASUREMENT_TYPE:
            raise ValueError(
                f"deployment records must be {MEASUREMENT_TYPE!r}, got {self.measurement_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra") or {}
        for key in SIMULATOR_KEYS:
            if key in extra:
                raise ValueError(f"refusing simulator field {key!r} on a MEASURED record")
        row.update(extra)
        row["measurement_type"] = MEASUREMENT_TYPE
        return row


def missing_required_fields(row: dict[str, Any]) -> list[str]:
    return [name for name in REQUIRED_REQUEST_FIELDS if name not in row]


def assert_measured_record(row: MappingLike) -> None:
    if not isinstance(row, dict):
        raise TypeError("record must be a dict")
    mt = row.get("measurement_type")
    if mt != MEASUREMENT_TYPE:
        raise ValueError(
            f"refusing non-MEASURED record (measurement_type={mt!r}); "
            "deployment_real must not mix simulator outputs"
        )
    overlap = SIMULATOR_KEYS.intersection(row)
    if overlap:
        raise ValueError(
            f"refusing record with simulator fields {sorted(overlap)}; "
            "never mix real deployment results with simulator outputs"
        )
    if mt in SIMULATOR_MEASUREMENT_TYPES:
        raise ValueError("simulator measurement_type is not allowed under deployment_real")


MappingLike = dict[str, Any]
