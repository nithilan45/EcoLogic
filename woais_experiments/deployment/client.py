"""HTTP client for the EcoLogic deployment endpoint.

Records client-observed end-to-end latency. Server spans (router, provider,
lifecycle) are copied from the JSON body when present. Malformed bodies are
logged, not guessed.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
import uuid
from typing import Any, Mapping
from urllib.parse import urljoin

from woais_experiments.deployment.records import (
    MEASUREMENT_TYPE,
    REQUIRED_REQUEST_FIELDS,
    RequestRecord,
    utc_timestamp,
)
from woais_experiments.latency.timing import ns_to_ms, now_ns

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class DeploymentClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        opener=None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._opener = opener or urllib.request.urlopen

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def health(self) -> dict[str, Any]:
        req = urllib.request.Request(self._url("health"), method="GET")
        with self._opener(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def infer(
        self,
        prompt: str,
        *,
        policy: str = "ecologic",
        request_id: str | None = None,
        forced_model: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> RequestRecord:
        rid = request_id or str(uuid.uuid4())
        payload: dict[str, Any] = {
            "prompt": prompt,
            "policy": policy,
            "request_id": rid,
            "measurement_type": MEASUREMENT_TYPE,
        }
        if forced_model is not None:
            payload["forced_model"] = forced_model
        if extra:
            payload.update(dict(extra))
        data = json.dumps(payload).encode("utf-8")
        last_status = 0
        last_error: str | None = None
        last_body: dict[str, Any] | None = None
        retry_count = 0
        t0 = now_ns()
        for attempt in range(self.max_retries + 1):
            retry_count = attempt
            try:
                req = urllib.request.Request(
                    self._url("v1/infer"),
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self._opener(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    last_status = int(getattr(resp, "status", 200) or 200)
                    last_body = _parse_json_object(raw)
                    if last_body is None:
                        last_error = "malformed_response"
                        break
                    if last_status >= 500 or last_status in RETRYABLE_STATUS:
                        last_error = "http_5xx"
                        if attempt < self.max_retries:
                            continue
                    else:
                        last_error = None if last_status < 400 else str(last_body.get("error_type") or "http_4xx")
                    break
            except urllib.error.HTTPError as exc:
                last_status = int(exc.code)
                raw = exc.read() if exc.fp is not None else b""
                last_body = _parse_json_object(raw)
                if last_body is None and raw:
                    last_error = "malformed_response"
                    if last_status < 500:
                        break
                else:
                    last_error = "http_5xx" if last_status >= 500 else "http_4xx"
                if last_status not in RETRYABLE_STATUS and last_status < 500:
                    break
            except TimeoutError:
                last_status = 0
                last_error = "timeout"
                last_body = None
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
                last_status = 0
                last_error = "connection_error"
                last_body = None
        t1 = now_ns()
        client_e2e = ns_to_ms(t1 - t0)
        return merge_client_record(
            request_id=rid,
            prompt=prompt,
            policy=policy,
            client_e2e_ms=client_e2e,
            retry_count=retry_count,
            http_status=last_status,
            error_type=last_error,
            body=last_body,
            extra=extra,
        )


def _parse_json_object(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def merge_client_record(
    *,
    request_id: str,
    prompt: str,
    policy: str,
    client_e2e_ms: float,
    retry_count: int,
    http_status: int,
    error_type: str | None,
    body: dict[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> RequestRecord:
    extra = dict(extra or {})
    if body is None:
        rec = RequestRecord(
            request_id=request_id,
            timestamp=utc_timestamp(),
            query_length_chars=len(prompt),
            input_tokens=None,
            router_decision_ms=None,
            provider_request_ms=None,
            time_to_first_token_ms=None,
            generation_ms=None,
            end_to_end_ms=client_e2e_ms,
            selected_model=None,
            output_tokens=None,
            realized_provider_cost=None,
            http_status=int(http_status),
            error_type=error_type or "malformed_response",
            retry_count=int(retry_count),
            policy=policy,
            setup=extra.get("setup"),
            setup_name=extra.get("setup_name"),
            concurrency=extra.get("concurrency"),
            prompt_id=extra.get("prompt_id"),
            generation_mode=extra.get("generation_mode"),
            backend=extra.get("backend"),
        )
        rec.extra["client_end_to_end_ms"] = client_e2e_ms
        rec.extra["server_body_ok"] = False
        return rec

    def _f(name: str) -> float | None:
        v = body.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _i(name: str) -> int | None:
        v = body.get(name)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    server_error = body.get("error_type")
    merged_error = error_type or (str(server_error) if server_error else None)
    rec = RequestRecord(
        request_id=str(body.get("request_id") or request_id),
        timestamp=str(body.get("timestamp") or utc_timestamp()),
        query_length_chars=int(body.get("query_length_chars") or len(prompt)),
        input_tokens=_i("input_tokens"),
        router_decision_ms=_f("router_decision_ms"),
        provider_request_ms=_f("provider_request_ms"),
        time_to_first_token_ms=_f("time_to_first_token_ms"),
        generation_ms=_f("generation_ms"),
        end_to_end_ms=client_e2e_ms,
        selected_model=None if body.get("selected_model") is None else str(body.get("selected_model")),
        output_tokens=_i("output_tokens"),
        realized_provider_cost=_f("realized_provider_cost"),
        http_status=int(body.get("http_status") or http_status),
        error_type=merged_error,
        retry_count=max(int(retry_count), int(body.get("retry_count") or 0)),
        lifecycle_state=body.get("lifecycle_state"),
        lifecycle_basis=body.get("lifecycle_basis"),
        process_id=_i("process_id"),
        request_index_in_process=_i("request_index_in_process"),
        process_uptime_s=_f("process_uptime_s"),
        setup=body.get("setup") or extra.get("setup"),
        setup_name=body.get("setup_name") or extra.get("setup_name"),
        policy=str(body.get("policy") or policy),
        concurrency=body.get("concurrency") if body.get("concurrency") is not None else extra.get("concurrency"),
        generation_mode=body.get("generation_mode") or extra.get("generation_mode"),
        backend=body.get("backend") or extra.get("backend"),
        prompt_id=body.get("prompt_id") or extra.get("prompt_id"),
        selected_tier=_i("selected_tier"),
        router_reason=body.get("router_reason"),
        input_tokens_source=body.get("input_tokens_source"),
        output_tokens_source=body.get("output_tokens_source"),
        cost_basis=body.get("cost_basis"),
        price_slug=body.get("price_slug"),
        price_source=body.get("price_source"),
        provider_name=body.get("provider_name"),
        paid_api=bool(body.get("paid_api", False)),
    )
    rec.extra["client_end_to_end_ms"] = client_e2e_ms
    rec.extra["server_end_to_end_ms"] = _f("end_to_end_ms")
    rec.extra["server_body_ok"] = True
    missing = [k for k in REQUIRED_REQUEST_FIELDS if k not in body]
    if missing:
        rec.extra["missing_server_fields"] = missing
        if rec.error_type is None and rec.http_status >= 200:
            rec.error_type = "malformed_response"
    return rec
