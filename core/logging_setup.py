"""
Structured JSON logging to stdout (Railway captures stdout).
Every log line is a single JSON object -> trivially filterable in Railway's
log explorer, e.g.  route:"orchestrator" AND layer:"rules"
"""
import json
import logging
import sys
import time

from core.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # merge structured extras (anything passed via logger.info(..., extra={"ctx": {...}}))
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            out.update(ctx)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    # quiet noisy libraries unless debugging
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if settings.LOG_LEVEL.upper() == "DEBUG" else logging.WARNING
        )


def log(logger: logging.Logger, level: int, msg: str, **ctx) -> None:
    """Convenience: log(logger, logging.INFO, "served", route="orch", layer="rules")"""
    logger.log(level, msg, extra={"ctx": ctx})
