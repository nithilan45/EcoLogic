"""JSON-lines structured logs plus a short human line on stderr."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", record.getMessage())
        fields = getattr(record, "fields", None) or {}
        extra = "".join(f" {k}={v}" for k, v in fields.items() if k != "event")
        return f"{record.levelname:<7} {event}{extra}"


def bind(logger: logging.Logger, **fields: Any) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"fields": fields})


class EventLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = dict(self.extra or {})
        user = kwargs.pop("extra", {}) or {}
        event = user.pop("event", None) or extra.get("event") or msg
        fields = dict(extra.get("fields") or {})
        fields.update(user.pop("fields", {}) or {})
        fields.update(user)
        kwargs["extra"] = {"event": event, "fields": fields}
        return msg, kwargs

    def event(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        self.log(level, event, extra={"event": event, "fields": fields})


def setup_logging(run_dir: Path, *, verbose: bool = False) -> EventLogger:
    logger = logging.getLogger("woais")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    json_path = run_dir / "run.log.jsonl"
    fh = logging.FileHandler(json_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonLinesFormatter())
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(HumanFormatter())
    logger.addHandler(sh)

    return EventLogger(logger, {})
