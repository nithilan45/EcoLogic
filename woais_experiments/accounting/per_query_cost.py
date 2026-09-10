"""Per-query realized LLM routing cost.

Prices are loaded from `woais_experiments/configs/models.json` only. This module
never invents a provider rate. Config stores USD per 1M tokens; per-token prices
are that figure divided by 1e6.

    realized_cost_i =
        input_tokens_i  * input_price_per_token(selected_model_i)
      + output_tokens_i * output_price_per_token(selected_model_i)
      + router_overhead_cost_i
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from woais_experiments.paths import CONFIGS

MILLION = 1_000_000.0
SCHEMA_VERSION = "1.0"


class UnknownPriceError(KeyError):
    """Model slug is not present in the committed price config."""


@dataclass(frozen=True)
class TokenPrices:
    model: str
    input_usd_per_million: float
    output_usd_per_million: float

    @property
    def input_price_per_token(self) -> float:
        return self.input_usd_per_million / MILLION

    @property
    def output_price_per_token(self) -> float:
        return self.output_usd_per_million / MILLION


@dataclass(frozen=True)
class RouterOverhead:
    name: str
    cost_usd: float
    tokens: float
    note: str = ""


@dataclass(frozen=True)
class QueryCostRecord:
    query_id: str
    selected_model: str
    input_tokens: int
    output_tokens: int
    input_price_per_token: float
    output_price_per_token: float
    router_overhead_cost: float
    router_overhead_tokens: float
    latency_ms: float | None = None

    def inference_cost(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )

    def realized_cost(self) -> float:
        return self.inference_cost() + self.router_overhead_cost

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["inference_cost"] = self.inference_cost()
        row["realized_cost"] = self.realized_cost()
        return row


class PriceTable:
    """Immutable map of model slug → committed USD/1M rates."""

    def __init__(self, prices: Mapping[str, TokenPrices]):
        self._prices = dict(prices)

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._prices))

    def require(self, model: str) -> TokenPrices:
        try:
            return self._prices[model]
        except KeyError as exc:
            known = ", ".join(self.models()) or "(none)"
            raise UnknownPriceError(
                f"no committed price for {model!r}; known models: {known}"
            ) from exc

    def inference_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.require(model)
        return (
            int(input_tokens) * p.input_price_per_token
            + int(output_tokens) * p.output_price_per_token
        )


def _as_million_rate(rates: Mapping[str, Any], kind: str) -> float:
    if kind == "input":
        keys = ("input", "in")
    else:
        keys = ("output", "out")
    for k in keys:
        if k in rates:
            return float(rates[k])
    raise KeyError(f"price block missing {kind} rate: {dict(rates)}")


def load_models_config() -> dict:
    return json.loads((CONFIGS / "models.json").read_text())


def load_price_table(cfg: dict | None = None) -> PriceTable:
    """Load every USD/1M block in the config (eval stack and RouteLLM)."""
    cfg = cfg if cfg is not None else load_models_config()
    rows: dict[str, TokenPrices] = {}
    for model, rates in cfg["usd_per_million"].items():
        rows[model] = TokenPrices(
            model=model,
            input_usd_per_million=_as_million_rate(rates, "input"),
            output_usd_per_million=_as_million_rate(rates, "output"),
        )
    for model, rates in cfg.get("routellm", {}).get("usd_per_million", {}).items():
        rows[model] = TokenPrices(
            model=model,
            input_usd_per_million=_as_million_rate(rates, "input"),
            output_usd_per_million=_as_million_rate(rates, "output"),
        )
    return PriceTable(rows)


def load_router_overhead(name: str = "none", cfg: dict | None = None) -> RouterOverhead:
    cfg = cfg if cfg is not None else load_models_config()
    block = cfg.get("router_overhead") or {}
    if name not in block:
        known = ", ".join(sorted(block)) or "(none)"
        raise KeyError(f"no router_overhead spec {name!r} in config; known: {known}")
    spec = block[name]
    return RouterOverhead(
        name=name,
        cost_usd=float(spec["cost_usd"]),
        tokens=float(spec["tokens"]),
        note=str(spec.get("note") or ""),
    )


def make_query(
    query_id: str,
    selected_model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: PriceTable,
    overhead: RouterOverhead | None = None,
    overhead_cost: float | None = None,
    overhead_tokens: float | None = None,
    latency_ms: float | None = None,
) -> QueryCostRecord:
    """Build a record, copying per-token prices from the committed table."""
    p = prices.require(selected_model)
    if overhead is not None:
        oh_cost = overhead.cost_usd if overhead_cost is None else float(overhead_cost)
        oh_tok = overhead.tokens if overhead_tokens is None else float(overhead_tokens)
    else:
        oh_cost = 0.0 if overhead_cost is None else float(overhead_cost)
        oh_tok = 0.0 if overhead_tokens is None else float(overhead_tokens)
    return QueryCostRecord(
        query_id=str(query_id),
        selected_model=selected_model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        input_price_per_token=p.input_price_per_token,
        output_price_per_token=p.output_price_per_token,
        router_overhead_cost=float(oh_cost),
        router_overhead_tokens=float(oh_tok),
        latency_ms=None if latency_ms is None else float(latency_ms),
    )


def realized_cost(record: QueryCostRecord) -> float:
    return record.realized_cost()


PER_QUERY_CSV_FIELDS = (
    "query_id",
    "selected_model",
    "input_tokens",
    "output_tokens",
    "input_price_per_token",
    "output_price_per_token",
    "router_overhead_cost",
    "router_overhead_tokens",
    "latency_ms",
    "inference_cost",
    "realized_cost",
)
