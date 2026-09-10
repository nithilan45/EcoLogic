"""Shared held-out protocol helpers. No test-set fitting lives here."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from io import StringIO
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.costs import TIERS
from woais_experiments.frozen import to_jsonable
from woais_experiments.paths import get_results_root

RESULTS_PREFIX = "heldout"
FROZEN_CONFIG_REL = f"{RESULTS_PREFIX}/frozen_config.json"
LOCK_REL = f"{RESULTS_PREFIX}/TEST_EVALUATED.lock"
FINAL_CSV_REL = f"{RESULTS_PREFIX}/final_test_results.csv"
QUERY_CSV_REL = f"{RESULTS_PREFIX}/query_level.csv"
N_BOOT = 10_000
N_PERM = 10_000
MODEL_SEED = 20260910


class HeldoutLeakageError(RuntimeError):
    """Train/val/test protocol was violated."""


class LockedExperimentError(RuntimeError):
    """Test already evaluated; refusing to change the frozen config."""


class TestQueryMismatchError(RuntimeError):
    """Router and baselines did not score the same test queries."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_source_sha256() -> str:
    """Hash of held-out protocol modules. Does not include result files."""
    from woais_experiments.paths import PACKAGE

    h = hashlib.sha256()
    root = PACKAGE / "heldout"
    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(sanitize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sanitize(obj: Any) -> Any:
    """JSON-stable values (numpy scalars → Python; non-finite → None)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return to_jsonable(obj)


def hash_config(config: Mapping[str, Any]) -> str:
    body = {k: v for k, v in config.items() if k not in {"config_sha256", "woais_run"}}
    return sha256_text(canonical(body))


def rows_to_csv(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> str:
    if not rows:
        return ""
    fields = list(fieldnames or rows[0].keys())
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({k: _csv_cell(row.get(k)) for k in fields})
    return buf.getvalue()


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def as_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=float)
    return np.asarray(x, dtype=float)


class ConstantClassifier:
    """Single-class fallback when the train oracle is degenerate."""

    def __init__(self, label: int) -> None:
        self.label = int(label)
        self.classes_ = np.array([self.label], dtype=int)

    def fit(self, x, y):  # noqa: ANN001
        return self

    def predict(self, x) -> np.ndarray:  # noqa: ANN001
        n = int(np.asarray(x).shape[0])
        return np.full(n, self.label, dtype=int)


def load_lock() -> dict[str, Any] | None:
    path = get_results_root() / LOCK_REL
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_if_locked(*, new_experiment: bool, action: str) -> dict[str, Any] | None:
    lock = load_lock()
    if lock and not new_experiment:
        raise LockedExperimentError(
            f"TEST_EVALUATED.lock exists; refusing to {action}. "
            "Pass --new-experiment to start a different held-out run. "
            "That flag does not un-see the previous test evaluation."
        )
    return lock


def mix_int(weights: Mapping[Any, float]) -> dict[int, float]:
    out = {int(t): 0.0 for t in TIERS}
    for k, v in weights.items():
        out[int(k)] = float(v)
    return out


def quota_assignment_sorted(ids: Sequence[str], mix: Mapping[int, float]) -> dict[str, int]:
    """Query-independent mix on sorted ids (largest remainder). No RNG."""
    ordered = sorted(str(i) for i in ids)
    n = len(ordered)
    mix_i = mix_int(mix)
    raw = [float(mix_i[t]) * n for t in TIERS]
    floors = [int(math.floor(x)) for x in raw]
    leftover = n - sum(floors)
    frac = sorted(enumerate([x - f for x, f in zip(raw, floors)]), key=lambda z: (-z[1], z[0]))
    counts = list(floors)
    for k in range(max(leftover, 0)):
        counts[frac[k % len(frac)][0]] += 1
    assign: dict[str, int] = {}
    cursor = 0
    for t, c in zip(TIERS, counts):
        for qid in ordered[cursor:cursor + c]:
            assign[qid] = int(t)
        cursor += c
    for qid in ordered:
        assign.setdefault(qid, int(TIERS[0]))
    return assign


def assert_same_test_population(
    assignments: Mapping[str, Mapping[str, int]],
    test_ids: Sequence[str],
) -> None:
    expected = [str(i) for i in test_ids]
    for name, assign in assignments.items():
        got = sorted(assign)
        if got != sorted(expected):
            raise TestQueryMismatchError(
                f"{name} scored {len(got)} queries; expected {len(expected)} identical test ids"
            )
        if set(got) != set(expected):
            raise TestQueryMismatchError(f"{name} test query set differs from the frozen test split")
