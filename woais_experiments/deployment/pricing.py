"""Deployment price table.

Product serving slugs in ``backend/main.py`` differ from eval-stack slugs in
``configs/models.json`` for tiers 1–2. This module never invents a rate:

- yaml ``prices_usd_per_million`` wins (operator-supplied)
- else yaml ``price_alias`` maps a serving slug onto a committed models.json rate
- else the serving slug must already exist in models.json

Every cost row records ``price_slug`` and ``price_source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from woais_experiments.accounting.per_query_cost import (
    PriceTable,
    TokenPrices,
    UnknownPriceError,
    load_price_table,
    load_router_overhead,
)


@dataclass(frozen=True)
class ResolvedPrice:
    serving_slug: str
    price_slug: str
    prices: TokenPrices
    source: str

    def inference_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            int(input_tokens) * self.prices.input_price_per_token
            + int(output_tokens) * self.prices.output_price_per_token
        )


class DeploymentPrices:
    def __init__(
        self,
        *,
        committed: PriceTable,
        overrides: Mapping[str, TokenPrices],
        aliases: Mapping[str, str],
        router_overhead_usd: float,
        router_overhead_name: str,
    ) -> None:
        self.committed = committed
        self.overrides = dict(overrides)
        self.aliases = dict(aliases)
        self.router_overhead_usd = float(router_overhead_usd)
        self.router_overhead_name = router_overhead_name

    def resolve(self, serving_slug: str) -> ResolvedPrice:
        if serving_slug in self.overrides:
            p = self.overrides[serving_slug]
            return ResolvedPrice(serving_slug, serving_slug, p, "deployment_config")
        if serving_slug in self.committed:
            p = self.committed.require(serving_slug)
            return ResolvedPrice(serving_slug, serving_slug, p, "models.json")
        alias = self.aliases.get(serving_slug)
        if alias:
            p = self.committed.require(alias)
            return ResolvedPrice(
                serving_slug,
                alias,
                p,
                f"eval_stack_proxy:{alias}",
            )
        known = ", ".join(sorted(set(self.overrides) | set(self.committed.models()) | set(self.aliases)))
        raise UnknownPriceError(
            f"no price for serving slug {serving_slug!r}; configure "
            f"prices_usd_per_million or price_alias. known: {known}"
        )

    def realized_provider_cost(
        self,
        serving_slug: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> tuple[float | None, ResolvedPrice | None]:
        if input_tokens is None or output_tokens is None:
            return None, None
        resolved = self.resolve(serving_slug)
        cost = resolved.inference_cost(int(input_tokens), int(output_tokens))
        cost += self.router_overhead_usd
        return cost, resolved


def _override_table(block: Mapping[str, Any] | None) -> dict[str, TokenPrices]:
    out: dict[str, TokenPrices] = {}
    if not block:
        return out
    for model, rates in block.items():
        if not isinstance(rates, Mapping):
            raise TypeError(f"price override for {model!r} must be a mapping")
        inn = rates.get("input", rates.get("in"))
        outv = rates.get("output", rates.get("out"))
        if inn is None or outv is None:
            raise KeyError(f"price override for {model!r} needs input and output")
        out[str(model)] = TokenPrices(
            model=str(model),
            input_usd_per_million=float(inn),
            output_usd_per_million=float(outv),
        )
    return out


def load_deployment_prices(cfg: Mapping[str, Any]) -> DeploymentPrices:
    committed = load_price_table()
    overhead_name = str(cfg.get("router_overhead_name") or "ecologic_keyword")
    overhead = load_router_overhead(overhead_name)
    aliases = {str(k): str(v) for k, v in (cfg.get("price_alias") or {}).items()}
    return DeploymentPrices(
        committed=committed,
        overrides=_override_table(cfg.get("prices_usd_per_million")),
        aliases=aliases,
        router_overhead_usd=float(overhead.cost_usd),
        router_overhead_name=overhead.name,
    )
