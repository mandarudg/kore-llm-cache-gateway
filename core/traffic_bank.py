"""
Traffic bank: append-only JSONL, one file per route per day, size-rotated.
This is the raw material for (a) rule-agreement analysis in shadow mode and
(b) fine-tuning exports (scripts/export_finetune_jsonl.py).

On Railway attach a Volume mounted at /data (or set TRAFFIC_BANK_DIR) --
container disk without a volume is wiped on every deploy.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict

from core.config import settings
from core.logging_setup import log

logger = logging.getLogger("traffic_bank")
_lock = threading.Lock()
_warned_disabled = False


def _path_for(route: str) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    base = os.path.join(settings.TRAFFIC_BANK_DIR, route)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{day}.jsonl")
    # size rotation
    try:
        if os.path.exists(path) and os.path.getsize(path) > settings.TRAFFIC_BANK_MAX_MB_PER_FILE * 1024 * 1024:
            n = 1
            while os.path.exists(f"{path}.{n}"):
                n += 1
            os.rename(path, f"{path}.{n}")
    except OSError:
        pass
    return path


def record(route: str, entry: Dict[str, Any]) -> None:
    """Fire-and-forget append. Never allowed to break the request path."""
    global _warned_disabled
    if not settings.TRAFFIC_BANK_ENABLED:
        return
    if settings.TRAFFIC_BANK_MODE == "slim":
        entry = {k: v for k, v in entry.items() if k not in ("request_body", "response_body")}
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock:
            with open(_path_for(route), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        if not _warned_disabled:
            log(logger, logging.ERROR,
                "traffic bank write failed -- check TRAFFIC_BANK_DIR volume mount",
                dir=settings.TRAFFIC_BANK_DIR)
            _warned_disabled = True
