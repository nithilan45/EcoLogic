"""Generation backends: dry-run stub vs paid chat completions.

Paid HTTP is used only when ``allow_api`` is true. Tokens are taken from
provider ``usage`` when present and left missing otherwise — never fabricated
from word counts.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from woais_experiments.latency.timing import InferenceTimer, ns_to_ms, now_ns

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_SYSTEM = (
    "Be concise and direct. Keep responses under 200 words unless more "
    "detail is specifically requested."
)


class PaidApiRefused(RuntimeError):
    """Raised when a paid provider is invoked without --allow-api."""


class RetryableProviderError(RuntimeError):
    def __init__(self, message: str, *, http_status: int = 0, error_type: str = "provider_error"):
        super().__init__(message)
        self.http_status = int(http_status)
        self.error_type = error_type


class MalformedProviderResponse(RetryableProviderError):
    def __init__(self, message: str, *, http_status: int = 200):
        super().__init__(message, http_status=http_status, error_type="malformed_response")


@dataclass
class ProviderResult:
    text: str | None
    http_status: int
    error_type: str | None
    input_tokens: int | None
    output_tokens: int | None
    input_tokens_source: str | None
    output_tokens_source: str | None
    time_to_first_token_ms: float | None = None
    generation_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _usage_tokens(payload: Mapping[str, Any]) -> tuple[int | None, int | None, str | None, str | None]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None, None
    inn = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    try:
        inn_i = int(inn) if inn is not None else None
    except (TypeError, ValueError):
        inn_i = None
    try:
        out_i = int(out) if out is not None else None
    except (TypeError, ValueError):
        out_i = None
    inn_src = "provider_usage" if inn_i is not None else None
    out_src = "provider_usage" if out_i is not None else None
    return inn_i, out_i, inn_src, out_src


def parse_chat_completion(payload: Any, *, http_status: int) -> ProviderResult:
    """Parse an OpenAI-style chat.completion object. Missing usage stays None."""
    if not isinstance(payload, Mapping):
        raise MalformedProviderResponse("provider body is not a JSON object", http_status=http_status)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MalformedProviderResponse("provider JSON missing choices", http_status=http_status)
    first = choices[0]
    if not isinstance(first, Mapping):
        raise MalformedProviderResponse("choices[0] is not an object", http_status=http_status)
    message = first.get("message")
    if isinstance(message, Mapping):
        text = message.get("content")
    else:
        text = first.get("text")
    if text is not None and not isinstance(text, str):
        raise MalformedProviderResponse("message content is not a string", http_status=http_status)
    inn, out, inn_src, out_src = _usage_tokens(payload)
    return ProviderResult(
        text=text,
        http_status=int(http_status),
        error_type=None,
        input_tokens=inn,
        output_tokens=out,
        input_tokens_source=inn_src,
        output_tokens_source=out_src,
    )


class StubProvider:
    """Local stub. No sockets. Tokens are labeled estimates, not provider usage."""

    def __init__(
        self,
        *,
        sleep_s: float = 0.001,
        output_tokens: int = 16,
        chars_per_token: float = 4.0,
    ) -> None:
        self.sleep_s = float(sleep_s)
        self.output_tokens = int(output_tokens)
        self.chars_per_token = float(chars_per_token)

    def complete(
        self,
        *,
        model: str,
        provider_name: str,
        prompt: str,
        stream: bool = False,
        timer: InferenceTimer | None = None,
    ) -> ProviderResult:
        del model, provider_name, stream
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        inn = max(1, int(len(prompt) / self.chars_per_token))
        return ProviderResult(
            text="[dry-run stub; no provider HTTP]",
            http_status=200,
            error_type=None,
            input_tokens=inn,
            output_tokens=self.output_tokens,
            input_tokens_source="chars_div_4_dry_run",
            output_tokens_source="stub_output_tokens",
            time_to_first_token_ms=None,
            generation_ms=None,
        )


def _endpoint_for(provider_name: str) -> str:
    name = provider_name.lower()
    if name in {"together", "togetherai", "together_ai"}:
        return TOGETHER_URL
    if name in {"openai", "oa"}:
        return OPENAI_URL
    raise PaidApiRefused(f"unknown provider {provider_name!r}")


def _api_key_for(provider_name: str) -> str:
    name = provider_name.lower()
    if name in {"together", "togetherai", "together_ai"}:
        key = os.environ.get("TOGETHER_API_KEY") or ""
        env = "TOGETHER_API_KEY"
    elif name in {"openai", "oa"}:
        key = os.environ.get("OPENAI_API_KEY") or ""
        env = "OPENAI_API_KEY"
    else:
        raise PaidApiRefused(f"unknown provider {provider_name!r}")
    if not key or key == "your-openai-key-here":
        raise PaidApiRefused(f"{env} is missing; cannot make a paid request")
    return key


class PaidChatProvider:
    def __init__(
        self,
        *,
        allow_api: bool,
        timeout_s: float = 30.0,
        max_tokens: int = 64,
        temperature: float = 0.0,
        system: str = DEFAULT_SYSTEM,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.allow_api = bool(allow_api)
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.system = system
        self._opener = opener or urllib.request.urlopen

    def complete(
        self,
        *,
        model: str,
        provider_name: str,
        prompt: str,
        stream: bool = False,
        timer: InferenceTimer | None = None,
    ) -> ProviderResult:
        if not self.allow_api:
            raise PaidApiRefused("paid provider refused: pass --allow-api")
        url = _endpoint_for(provider_name)
        key = _api_key_for(provider_name)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": bool(stream),
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        if stream:
            return self._complete_stream(req, timer=timer)
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                status = int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read() if exc.fp is not None else b""
            err = f"provider HTTP {status}"
            if status == 429:
                raise RetryableProviderError(err, http_status=status, error_type="http_429") from exc
            if status >= 500 or status == 408:
                etype = "timeout" if status == 408 else "http_5xx"
                raise RetryableProviderError(err, http_status=status, error_type=etype) from exc
            raise RetryableProviderError(err, http_status=status, error_type="http_4xx") from exc
        except TimeoutError as exc:
            raise RetryableProviderError("provider timeout", http_status=0, error_type="timeout") from exc
        except urllib.error.URLError as exc:
            raise RetryableProviderError(
                f"provider connection error: {exc.reason}",
                http_status=0,
                error_type="connection_error",
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedProviderResponse("provider body is not JSON", http_status=status) from exc
        return parse_chat_completion(payload, http_status=status)

    def _complete_stream(
        self,
        req: urllib.request.Request,
        *,
        timer: InferenceTimer | None,
    ) -> ProviderResult:
        chunks: list[str] = []
        first_ns: int | None = None
        last_ns: int | None = None
        usage_payload: dict[str, Any] = {}
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                if status >= 400:
                    raw = resp.read()
                    raise RetryableProviderError(
                        f"provider HTTP {status}",
                        http_status=status,
                        error_type="http_5xx" if status >= 500 else "http_4xx",
                    )
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith("data:"):
                        continue
                    data_str = text[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if isinstance(payload.get("usage"), dict):
                        usage_payload = payload
                    choices = payload.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        now = now_ns()
                        if first_ns is None:
                            first_ns = now
                            if timer is not None:
                                timer.note_first_token()
                        last_ns = now
                        chunks.append(piece)
                if timer is not None and chunks:
                    timer.note_last_token()
        except TimeoutError as exc:
            raise RetryableProviderError("provider timeout", http_status=0, error_type="timeout") from exc
        except urllib.error.URLError as exc:
            raise RetryableProviderError(
                f"provider connection error: {exc.reason}",
                http_status=0,
                error_type="connection_error",
            ) from exc
        inn, out, inn_src, out_src = _usage_tokens(usage_payload)
        ttft = None
        gen = None
        if timer is not None and timer.has("request_start") and first_ns is not None:
            ttft = ns_to_ms(first_ns - timer.marks["request_start"])
        if first_ns is not None and last_ns is not None:
            gen = ns_to_ms(last_ns - first_ns)
        return ProviderResult(
            text="".join(chunks) if chunks else None,
            http_status=200,
            error_type=None if chunks else "malformed_response",
            input_tokens=inn,
            output_tokens=out,
            input_tokens_source=inn_src,
            output_tokens_source=out_src,
            time_to_first_token_ms=ttft,
            generation_ms=gen,
        )


def make_provider(cfg: Mapping[str, Any], *, allow_api: bool, dry_run: bool) -> StubProvider | PaidChatProvider:
    stub_cfg = cfg.get("stub") or {}
    if dry_run or not allow_api:
        return StubProvider(
            sleep_s=float(stub_cfg.get("sleep_s", 0.001)),
            output_tokens=int(stub_cfg.get("output_tokens", 16)),
            chars_per_token=float(stub_cfg.get("chars_per_token", 4.0)),
        )
    paid = cfg.get("paid") or {}
    return PaidChatProvider(
        allow_api=True,
        timeout_s=float(paid.get("timeout_s", 30.0)),
        max_tokens=int(paid.get("max_tokens", 64)),
        temperature=float(paid.get("temperature", 0.0)),
        system=str(paid.get("system", DEFAULT_SYSTEM)),
    )
