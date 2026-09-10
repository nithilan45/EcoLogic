"""Minimal HTTP inference endpoint wrapping the EcoLogic router.

Stdlib ``http.server`` so tests and Docker do not need uvicorn/FastAPI.
Does not provision cloud infrastructure. Lambda-compatible handler included.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import AppContext, build_context, handle_inference
from woais_experiments.deployment.lifecycle import platform_cloud_backend
from woais_experiments.deployment.records import v2_measurement_type

HEALTH_PATHS = {"/health", "/healthz"}
LIFECYCLE_PATHS = {"/lifecycle"}
INFER_PATHS = {"/infer", "/v1/infer"}


class InferHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, context: AppContext):
        super().__init__(server_address, InferHandler)
        self.context = context


class InferHandler(BaseHTTPRequestHandler):
    server_version = "EcoLogicDeploy/1.0"

    @property
    def ctx(self) -> AppContext:
        return self.server.context  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if (self.ctx.config or {}).get("access_log"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        payload.setdefault("measurement_type", self.ctx.measurement_type)
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in HEALTH_PATHS or path == "/":
            life = self.ctx.lifecycle
            self._write_json(200, {
                "ok": True,
                "measurement_type": self.ctx.measurement_type,
                "backend": self.ctx.backend,
                "generation_mode": self.ctx.generation_mode,
                "allow_api": self.ctx.allow_api,
                "dry_run": self.ctx.dry_run,
                "process_id": life.process_id,
                "request_count": life.request_count,
                "paid_api": bool(self.ctx.allow_api and not self.ctx.dry_run),
            })
            return
        if path in LIFECYCLE_PATHS:
            life = self.ctx.lifecycle
            self._write_json(200, {
                "measurement_type": self.ctx.measurement_type,
                "process_id": life.process_id,
                "request_count": life.request_count,
                "process_start_monotonic": life.process_start_monotonic,
                "init_type": life.init_type,
                "note": (
                    "cold/warm is assigned on /infer from process request index; "
                    "idle time does not relabel a live process as cold"
                ),
            })
            return
        self._write_json(404, {"error_type": "not_found", "measurement_type": self.ctx.measurement_type})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in INFER_PATHS:
            self._write_json(404, {"error_type": "not_found", "measurement_type": self.ctx.measurement_type})
            return
        length_s = self.headers.get("Content-Length")
        raw_ok = True
        body: dict[str, Any] | None = None
        try:
            n = int(length_s) if length_s else 0
        except ValueError:
            n = 0
            raw_ok = False
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            raw_ok = False
        else:
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    body = parsed
                else:
                    raw_ok = False
            except (UnicodeDecodeError, json.JSONDecodeError):
                raw_ok = False
        status, rec = handle_inference(body, ctx=self.ctx, raw_ok=raw_ok)
        self._write_json(status, rec.to_dict())


def serve_forever(server: InferHTTPServer) -> None:
    server.serve_forever()


def start_server(
    ctx: AppContext,
    *,
    host: str,
    port: int,
) -> tuple[InferHTTPServer, threading.Thread]:
    httpd = InferHTTPServer((host, int(port)), ctx)
    thread = threading.Thread(target=httpd.serve_forever, name="ecologic-deploy", daemon=True)
    thread.start()
    return httpd, thread


def stop_server(httpd: InferHTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


def _event_body(event: MappingLike) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(event, dict):
        return None, False
    if "prompt" in event or "policy" in event:
        return event, True
    raw = event.get("body")
    if raw is None:
        return None, False
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw)
        except (ValueError, TypeError):
            return None, False
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None, False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    return parsed, True


MappingLike = dict[str, Any]

_LAMBDA_CTX: AppContext | None = None


def reset_lambda_context() -> None:
    global _LAMBDA_CTX
    _LAMBDA_CTX = None


def _lambda_context() -> AppContext:
    global _LAMBDA_CTX
    if _LAMBDA_CTX is None:
        allow = os.environ.get("ECOLOGIC_DEPLOY_ALLOW_API", "").strip() in {"1", "true", "TRUE", "yes"}
        allow_cloud = os.environ.get("ECOLOGIC_DEPLOY_ALLOW_CLOUD", "").strip() in {"1", "true", "TRUE", "yes"}
        dry = not allow
        cfg = load_deployment_config(os.environ.get("ECOLOGIC_DEPLOY_CONFIG"))
        paid = bool(allow) and not dry
        serverless = bool(paid and allow_cloud and platform_cloud_backend())
        _LAMBDA_CTX = build_context(
            cfg,
            allow_api=allow,
            dry_run=dry,
            backend="lambda",
            allow_cloud=allow_cloud,
            measurement_type=v2_measurement_type(paid=paid, serverless=serverless),
        )
    return _LAMBDA_CTX


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """AWS Lambda / Function URL compatible wrapper. Does not create AWS resources."""
    del context
    body, raw_ok = _event_body(event or {})
    status, rec = handle_inference(body, ctx=_lambda_context(), raw_ok=raw_ok)
    payload = rec.to_dict()
    return {
        "statusCode": int(status),
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, default=str),
        "measurement_type": rec.measurement_type,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EcoLogic deployment HTTP endpoint")
    p.add_argument("--config", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--backend", default="local", choices=("local", "docker", "cloudrun", "lambda"))
    p.add_argument("--allow-api", action="store_true", help="Permit paid provider calls")
    p.add_argument("--allow-cloud", action="store_true", help="Permit serverless labels on Cloud Run / Lambda")
    p.add_argument("--dry-run", action="store_true", help="Stub provider; no paid HTTP")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_api and args.dry_run:
        print("error: --allow-api and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    dry_run = not args.allow_api
    allow_cloud = bool(args.allow_cloud) or os.environ.get("ECOLOGIC_DEPLOY_ALLOW_CLOUD", "").strip() in {
        "1", "true", "TRUE", "yes",
    }
    cfg = load_deployment_config(args.config)
    listen = cfg.get("listen") or {}
    host = args.host or os.environ.get("HOST") or str(listen.get("host") or "0.0.0.0")
    port = int(args.port or os.environ.get("PORT") or listen.get("port") or 8080)
    paid = bool(args.allow_api) and not dry_run
    serverless = bool(paid and allow_cloud and platform_cloud_backend())
    measurement_type = v2_measurement_type(paid=paid, serverless=serverless)
    ctx = build_context(
        cfg,
        allow_api=bool(args.allow_api),
        dry_run=dry_run,
        backend=str(args.backend),
        allow_cloud=allow_cloud,
        measurement_type=measurement_type,
    )
    httpd = InferHTTPServer((host, port), ctx)
    bound = httpd.server_address
    print(
        f"EcoLogic deploy listening on http://{bound[0]}:{bound[1]} "
        f"mode={ctx.generation_mode} measurement_type={ctx.measurement_type} "
        f"serverless={ctx.serverless_labeled}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server(httpd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
