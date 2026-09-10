"""Summarize the frozen held-out test evaluation. Does not retune."""

from __future__ import annotations

import argparse
import csv
from typing import Any, Mapping, Sequence

from woais_experiments.heldout.build_split import load_manifest
from woais_experiments.heldout.common import FINAL_CSV_REL
from woais_experiments.paths import get_results_root

TABLE_METHODS = (
    "always_cheap",
    "always_strong",
    "random_matched",
    "cost_matched_static",
    "logistic",
    "tree",
    "threshold",
    "ecologic_heuristic",
    "oracle",
)

TABLE_FIELDS = (
    "method",
    "quality",
    "realized_cost",
    "quality_advantage_vs_cost_matched_static",
    "oracle_gap",
)


def load_final_rows(path=None) -> list[dict[str, Any]]:
    p = path or (get_results_root() / FINAL_CSV_REL)
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def routing_claim(row: Mapping[str, Any]) -> str:
    """Conservative: do not claim routing helps unless the interval and test agree."""
    adv = _f(row, "quality_advantage_vs_cost_matched_static")
    lo = _f(row, "advantage_ci_lo")
    hi = _f(row, "advantage_ci_hi")
    p = _f(row, "advantage_p_raw")
    if adv is None:
        return "no held-out advantage estimate; no routing-helps claim"
    if lo is not None and lo > 0 and p is not None and p < 0.05:
        return (
            "held-out evidence supports a quality advantage vs the pre-registered "
            "cost-matched static mix"
        )
    if adv > 0:
        return (
            "point estimate is positive vs cost-matched static, but the held-out "
            "interval/test do not support a routing-helps claim"
        )
    if adv < 0:
        return (
            "held-out result does not show a quality advantage vs cost-matched static; "
            "the loss is kept (no post-test tuning)"
        )
    return "held-out quality matches cost-matched static; no routing-helps claim"


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    by_name = {r["method"]: r for r in rows}
    headers = TABLE_FIELDS
    widths = {h: len(h) for h in headers}
    rendered: list[dict[str, str]] = []
    for name in TABLE_METHODS:
        if name not in by_name:
            continue
        r = by_name[name]
        cells = {"method": name}
        for h in headers[1:]:
            v = _f(r, h)
            cells[h] = "" if v is None else f"{v:.6g}"
        rendered.append(cells)
        for h in headers:
            widths[h] = max(widths[h], len(cells[h]))
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for cells in rendered:
        lines.append("  ".join(cells[h].ljust(widths[h]) for h in headers))
    return "\n".join(lines)


def print_summary(payload: Mapping[str, Any] | None = None) -> None:
    if payload is None:
        manifest = load_manifest()
        n = manifest["n"]
        rows = load_final_rows()
        selected = next((r["method"] for r in rows if r.get("selected") in {"1", "true", "True"}), None)
    else:
        n = payload["n"]
        rows = payload["rows"]
        selected = payload.get("selected_method")
    print(f"TRAIN n\t{n['train']}")
    print(f"VAL n\t{n['val']}")
    print(f"TEST n\t{n['test']}")
    print()
    print(format_table(rows))
    print()
    by_name = {r["method"]: r for r in rows}
    if selected and selected in by_name:
        print(f"selected_method\t{selected}")
        print(f"claim\t{routing_claim(by_name[selected])}")
    print("note\tStage 1–2 n=364 is historical and is not this test set.")
    print("note\tquality_advantage_vs_cost_matched_static is vs the VAL-frozen static mix on TEST.")
    print("note\toracle is a hindsight upper bound, not a deployable router.")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Print the frozen held-out test summary.")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
