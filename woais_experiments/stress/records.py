"""Request-log helpers for the stress suite.

MEASURED records reuse the deployment schema and must never carry simulator
keys. SIMULATED records are tagged separately and written to a different tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from woais_experiments.deployment.records import (
    MEASUREMENT_TYPE as DEPLOY_MEASURED,
    SIMULATOR_KEYS,
    assert_measured_record,
)
from woais_experiments.frozen import to_jsonable
from woais_experiments.paths import assert_inside_results, get_results_root
from woais_experiments.stress import MEASURED, SIMULATED

REQUIRED_STRESS_FIELDS = (
    "measurement_type",
    "request_id",
    "policy",
    "stress_profile",
    "load_multiplier",
    "scheduled_arrival_s",
)


class MixedSourceError(ValueError):
    """MEASURED and SIMULATED rows were combined, or a record is mis-tagged."""


def cell_key(profile: str, multiplier: float, policy: str) -> str:
    return f"{profile}|{float(multiplier):g}|{policy}"


def results_prefix(measurement_type: str) -> str:
    mt = str(measurement_type)
    if mt == MEASURED:
        return "stress/measured"
    if mt == SIMULATED:
        return "stress/simulated"
    raise MixedSourceError(f"unknown measurement_type {mt!r}")


def cell_jsonl_relpath(measurement_type: str, profile: str, multiplier: float, policy: str) -> str:
    prefix = results_prefix(measurement_type)
    return f"{prefix}/requests/{profile}__{float(multiplier):g}x__{policy}.jsonl"


def assert_stress_record(row: Mapping[str, Any], *, expected: str) -> None:
    if not isinstance(row, dict):
        raise TypeError("record must be a dict")
    mt = row.get("measurement_type")
    if mt != expected:
        raise MixedSourceError(
            f"refusing record with measurement_type={mt!r}; expected {expected!r}"
        )
    missing = [k for k in REQUIRED_STRESS_FIELDS if k not in row]
    if missing:
        raise MixedSourceError(f"stress record missing {missing}")
    if expected == MEASURED:
        overlap = SIMULATOR_KEYS.intersection(row)
        if overlap:
            raise MixedSourceError(
                f"MEASURED stress record has simulator fields {sorted(overlap)}"
            )
        assert_measured_record({**row, "measurement_type": DEPLOY_MEASURED})
        if row.get("suite") not in (None, "stress_measured"):
            raise MixedSourceError(f"unexpected suite={row.get('suite')!r} on MEASURED record")
    elif expected == SIMULATED:
        if mt != SIMULATED:
            raise MixedSourceError("SIMULATED records must set measurement_type='SIMULATED'")
        if row.get("suite") not in (None, "stress_simulated"):
            raise MixedSourceError(f"unexpected suite={row.get('suite')!r} on SIMULATED record")
    else:
        raise MixedSourceError(f"unknown expected type {expected!r}")


def assert_homogeneous(rows: Sequence[Mapping[str, Any]], expected: str) -> None:
    if not rows:
        return
    types = {r.get("measurement_type") for r in rows}
    if types != {expected}:
        raise MixedSourceError(
            f"refusing mixed measurement_type values {sorted(repr(t) for t in types)}; "
            f"expected only {expected!r}"
        )
    for row in rows:
        assert_stress_record(row, expected=expected)


def dumps_row(row: Mapping[str, Any]) -> str:
    return json.dumps(to_jsonable(dict(row)), default=str, allow_nan=False)


def append_jsonl(relpath: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    dest = assert_inside_results(get_results_root() / relpath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(dumps_row(row) + "\n")
    return dest


def write_jsonl(relpath: str, rows: Iterable[Mapping[str, Any]], *, clobber: bool = False) -> Path:
    dest = assert_inside_results(get_results_root() / relpath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not clobber:
        raise FileExistsError(f"refusing to overwrite {dest}; pass resume or clobber")
    payload = "".join(dumps_row(row) + "\n" for row in rows)
    dest.write_text(payload, encoding="utf-8")
    return dest


def load_jsonl(path: Path, *, expected: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise MixedSourceError(f"jsonl line in {path} is not an object")
        rows.append(row)
    if expected is not None:
        assert_homogeneous(rows, expected)
    return rows


def occupancy_at_arrivals(
    arrivals_s: Sequence[float],
    completes_s: Sequence[float | None],
) -> list[int]:
    """In-system count immediately before each arrival (the arriving request is excluded)."""
    if len(arrivals_s) != len(completes_s):
        raise ValueError("arrivals and completes length mismatch")
    events: list[tuple[float, int, int]] = []
    for i, a in enumerate(arrivals_s):
        events.append((float(a), 1, i))  # arrival after completions at the same instant
    for i, c in enumerate(completes_s):
        if c is None:
            continue
        events.append((float(c), 0, i))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    occ = 0
    at_arrival = [0] * len(arrivals_s)
    for _t, kind, i in events:
        if kind == 0:
            occ = max(0, occ - 1)
        else:
            at_arrival[i] = occ
            occ += 1
    return at_arrival
