"""In-process inference used by the HTTP app, Lambda handler, and tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from woais_experiments.deployment.lifecycle import ProcessLifecycle
from woais_experiments.deployment.pricing import DeploymentPrices, load_deployment_prices
from woais_experiments.deployment.provider import (
    MalformedProviderResponse,
    PaidApiRefused,
    ProviderResult,
    RetryableProviderError,
    make_provider,
)
from woais_experiments.deployment.records import RequestRecord, utc_timestamp
from woais_experiments.deployment.router import classify_prompt, load_production_router, model_for_tier
from woais_experiments.latency.timing import InferenceTimer


class InferError(ValueError):
    def __init__(self, message: str, *, http_status: int = 400, error_type: str = "malformed_request"):
        super().__init__(message)
        self.http_status = http_status
        self.error_type = error_type


@dataclass
class AppContext:
    allow_api: bool
    dry_run: bool
    config: dict[str, Any]
    lifecycle: ProcessLifecycle
    prices: DeploymentPrices
    provider: Any
    models: dict[str, Any]
    classify_fn: Callable[[str], Any]
    max_retries: int = 2
    backend: str = "local"
    roles: dict[str, str] = field(default_factory=lambda: {"cheap": "tier1", "strong": "tier3"})

    @property
    def generation_mode(self) -> str:
        return "paid_api" if self.allow_api and not self.dry_run else "dry_run"

    @property
    def stream(self) -> bool:
        return bool((self.config.get("paid") or {}).get("stream", False)) and not self.dry_run

    @property
    def environment_kind(self) -> str:
        if self.dry_run or not self.allow_api:
            if self.backend == "lambda":
                return "in_process_lambda_stub"
            return "local_stub"
        if self.backend == "lambda":
            return "in_process_lambda"
        return "remote_http"

    @property
    def cloud_measured(self) -> bool:
        return bool(
            self.allow_api
            and not self.dry_run
            and self.backend in {"http", "cloud_run", "remote"}
        )


def _role_cfg(ctx: AppContext, role: str) -> dict[str, Any]:
    key = str(ctx.roles.get(role) or role)
    if key in ctx.models:
        return ctx.models[key]
    if key.startswith("tier"):
        return model_for_tier(ctx.models, int(key.replace("tier", "")))
    raise InferError(f"unknown role {role!r}", http_status=500, error_type="config_error")


def _select_model(
    ctx: AppContext,
    *,
    policy: str,
    prompt: str,
    forced_model: str | None,
    timer: InferenceTimer,
) -> tuple[str, str, int | None, str | None, float | None]:
    """Return selected_model, provider, tier, reason, router_decision_ms."""
    if policy in {"direct_cheap", "direct_strong"}:
        role = "cheap" if policy == "direct_cheap" else "strong"
        cfg = _role_cfg(ctx, role)
        return str(cfg["name"]), str(cfg["provider"]), int(str(ctx.roles[role]).replace("tier", "")), None, None
    if policy == "static_mixture":
        if not forced_model:
            raise InferError("static_mixture requires forced_model", error_type="malformed_request")
        provider = _provider_for_model(ctx, forced_model)
        return forced_model, provider, None, "static_mixture", None
    if policy != "ecologic":
        raise InferError(f"unknown policy {policy!r}")
    timer.mark("router_start")
    decision = classify_prompt(prompt, models=ctx.models, classify_fn=ctx.classify_fn)
    timer.mark("router_end")
    router_ms = timer.span("router_start", "router_end")
    return decision.selected_model, decision.provider, decision.recommended_tier, decision.reason, router_ms


def _provider_for_model(ctx: AppContext, model: str) -> str:
    for cfg in ctx.models.values():
        if isinstance(cfg, Mapping) and str(cfg.get("name")) == model:
            return str(cfg.get("provider") or "together")
    raise InferError(f"forced_model {model!r} is not in the serving catalog")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, PaidApiRefused):
        return False
    if isinstance(exc, InferError):
        return False
    if isinstance(exc, MalformedProviderResponse):
        return False
    if isinstance(exc, RetryableProviderError):
        return (
            exc.error_type in {
                "http_5xx", "http_429", "timeout", "connection_error", "provider_error",
            }
            or exc.http_status >= 500
            or exc.http_status == 429
        )
    return False


def call_with_retries(
    fn: Callable[[], ProviderResult],
    *,
    max_retries: int,
) -> tuple[ProviderResult | None, int, str | None, int]:
    """Run ``fn`` up to 1 + max_retries times. Returns result, retry_count, error_type, http_status."""
    last_err: BaseException | None = None
    retries = 0
    for attempt in range(int(max_retries) + 1):
        try:
            result = fn()
            return result, attempt, None, int(result.http_status)
        except Exception as exc:  # noqa: BLE001 — recorded onto the request log
            last_err = exc
            retries = attempt
            if attempt >= int(max_retries) or not _retryable(exc):
                break
    status = 0
    err_type = "unknown"
    if isinstance(last_err, RetryableProviderError):
        status = last_err.http_status
        err_type = last_err.error_type
    elif isinstance(last_err, PaidApiRefused):
        status = 403
        err_type = "paid_api_refused"
    elif last_err is not None:
        err_type = type(last_err).__name__
    return None, retries, err_type, status


def _empty_record(*, request_id: str, prompt: str, policy: str, ctx: AppContext) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=utc_timestamp(),
        query_length_chars=len(prompt),
        input_tokens=None,
        router_decision_ms=None,
        provider_request_ms=None,
        time_to_first_token_ms=None,
        generation_ms=None,
        end_to_end_ms=None,
        selected_model=None,
        output_tokens=None,
        realized_provider_cost=None,
        http_status=0,
        error_type=None,
        retry_count=0,
        generation_mode=ctx.generation_mode,
        backend=ctx.backend,
        environment_kind=ctx.environment_kind,
        cloud_measured=ctx.cloud_measured,
        cost_scope="final_attempt_only",
        policy=policy,
        paid_api=bool(ctx.allow_api and not ctx.dry_run),
    )


def handle_inference(
    body: Mapping[str, Any] | None,
    *,
    ctx: AppContext,
    raw_ok: bool = True,
) -> tuple[int, RequestRecord]:
    """Classify (optional), call the provider, and return ``(http_status, record)``."""
    timer = InferenceTimer()
    timer.mark("request_start")
    life = ctx.lifecycle.observe()

    if not raw_ok:
        rec = _empty_record(request_id=str(uuid.uuid4()), prompt="", policy="", ctx=ctx)
        rec.http_status = 400
        rec.error_type = "malformed_request"
        rec.lifecycle_state = life.lifecycle_state
        rec.lifecycle_basis = life.lifecycle_basis
        rec.process_id = life.process_id
        rec.request_index_in_process = life.request_index_in_process
        rec.process_uptime_s = life.process_uptime_s
        timer.mark("response_end")
        rec.end_to_end_ms = timer.span("request_start", "response_end")
        return 400, rec

    body = body or {}
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        rec = _empty_record(
            request_id=str(body.get("request_id") or uuid.uuid4()),
            prompt="" if not isinstance(prompt, str) else prompt,
            policy=str(body.get("policy") or ""),
            ctx=ctx,
        )
        rec.http_status = 400
        rec.error_type = "malformed_request"
        rec.lifecycle_state = life.lifecycle_state
        rec.lifecycle_basis = life.lifecycle_basis
        rec.process_id = life.process_id
        rec.request_index_in_process = life.request_index_in_process
        rec.process_uptime_s = life.process_uptime_s
        timer.mark("response_end")
        rec.end_to_end_ms = timer.span("request_start", "response_end")
        return 400, rec

    prompt = prompt.strip()
    policy = str(body.get("policy") or "ecologic")
    request_id = str(body.get("request_id") or uuid.uuid4())
    forced = body.get("forced_model")
    if forced is not None:
        forced = str(forced)

    rec = _empty_record(request_id=request_id, prompt=prompt, policy=policy, ctx=ctx)
    rec.setup = body.get("setup")
    rec.setup_name = body.get("setup_name")
    rec.concurrency = body.get("concurrency")
    rec.prompt_id = body.get("prompt_id")
    rec.lifecycle_state = life.lifecycle_state
    rec.lifecycle_basis = life.lifecycle_basis
    rec.process_id = life.process_id
    rec.request_index_in_process = life.request_index_in_process
    rec.process_uptime_s = life.process_uptime_s

    try:
        model, provider_name, tier, reason, router_ms = _select_model(
            ctx, policy=policy, prompt=prompt, forced_model=forced, timer=timer
        )
    except InferError as exc:
        rec.http_status = exc.http_status
        rec.error_type = exc.error_type
        timer.mark("response_end")
        rec.end_to_end_ms = timer.span("request_start", "response_end")
        rec.router_decision_ms = timer.span("router_start", "router_end")
        return rec.http_status, rec

    rec.selected_model = model
    rec.provider_name = provider_name
    rec.selected_tier = tier
    rec.router_reason = reason
    rec.router_decision_ms = router_ms
    timer.mark("provider_start")

    def _once() -> ProviderResult:
        return ctx.provider.complete(
            model=model,
            provider_name=provider_name,
            prompt=prompt,
            stream=ctx.stream,
            timer=timer,
        )

    result, retry_count, err_type, status = call_with_retries(_once, max_retries=ctx.max_retries)
    rec.retry_count = int(retry_count)
    timer.mark("response_end")
    rec.provider_request_ms = timer.span("provider_start", "response_end")
    rec.end_to_end_ms = timer.span("request_start", "response_end")

    if result is None:
        rec.http_status = int(status) or 502
        rec.error_type = err_type or "provider_error"
        if rec.error_type and rec.http_status < 400:
            rec.http_status = 502
        return rec.http_status, rec

    rec.http_status = int(result.http_status)
    rec.error_type = result.error_type or err_type
    rec.input_tokens = result.input_tokens
    rec.output_tokens = result.output_tokens
    rec.input_tokens_source = result.input_tokens_source
    rec.output_tokens_source = result.output_tokens_source
    rec.time_to_first_token_ms = result.time_to_first_token_ms
    rec.generation_ms = result.generation_ms
    if result.extra:
        rec.extra.update(result.extra)

    if rec.selected_model and rec.input_tokens is not None and rec.output_tokens is not None:
        cost, resolved = ctx.prices.realized_provider_cost(
            rec.selected_model, rec.input_tokens, rec.output_tokens
        )
        rec.realized_provider_cost = cost
        if resolved is not None:
            rec.price_slug = resolved.price_slug
            rec.price_source = resolved.source
            rec.cost_basis = (
                "dry_run_token_estimate"
                if rec.input_tokens_source == "chars_div_4_dry_run"
                else "usage_times_price_table"
            )

    ok = rec.http_status < 400 and rec.error_type is None
    if not ok and rec.http_status < 400:
        rec.http_status = 502
    return rec.http_status, rec


def load_serving_models(cfg: Mapping[str, Any]) -> dict[str, Any]:
    production_fn, production_models = load_production_router()
    del production_fn
    override = cfg.get("models")
    if not override:
        return dict(production_models)
    out = dict(production_models)
    for key, spec in override.items():
        if not isinstance(spec, Mapping):
            continue
        row = dict(out.get(key) or {})
        row.update(spec)
        out[str(key)] = row
    return out


def build_context(
    cfg: Mapping[str, Any],
    *,
    allow_api: bool,
    dry_run: bool,
    backend: str = "local",
    lifecycle: ProcessLifecycle | None = None,
    provider: Any = None,
    classify_fn: Callable[[str], Any] | None = None,
    models: dict[str, Any] | None = None,
) -> AppContext:
    loaded_fn, _production = load_production_router()
    serving = models if models is not None else load_serving_models(cfg)
    prices = load_deployment_prices(cfg)
    if provider is None:
        provider = make_provider(cfg, allow_api=bool(allow_api), dry_run=bool(dry_run))
    roles = dict(cfg.get("roles") or {"cheap": "tier1", "strong": "tier3"})
    retries = int((cfg.get("retries") or {}).get("max_retries", 2))
    return AppContext(
        allow_api=bool(allow_api),
        dry_run=bool(dry_run),
        config=dict(cfg),
        lifecycle=lifecycle or ProcessLifecycle(),
        prices=prices,
        provider=provider,
        models=serving,
        classify_fn=classify_fn or loaded_fn,
        max_retries=retries,
        backend=str(backend),
        roles=roles,
    )
