"""
Exact-match response cache for the /search (RAG answer generation) route.

Key = sha256( normalized user query + chunk IDs + channel ), so:
  - the same question phrased identically hits the cache,
  - a KB re-index (different chunk IDs) misses -> no stale answers,
  - digital vs voice answers never cross-contaminate.

Backends: in-memory LRU (default) or Redis if REDIS_URL is set.
Semantic (embedding) matching is a Phase-2 extension; the key derivation here
is already namespaced so it can be added without invalidating anything.
"""
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from core.config import settings
from core.logging_setup import log

logger = logging.getLogger("cache")

_CHUNK_ID_RE = re.compile(r"chunk_id:\s*([\w\-]+)", re.IGNORECASE)
_QUERY_RE = re.compile(r"Query:\s*(.+?)(?:$|\n)", re.IGNORECASE | re.DOTALL)


def derive_key(user_text: str, sys_txt: str) -> Optional[str]:
    """Build the cache key from an answer-gen request; None => don't cache."""
    chunk_ids = _CHUNK_ID_RE.findall(user_text)
    qm = _QUERY_RE.search(user_text)
    query = (qm.group(1) if qm else user_text)[-500:]
    query_norm = re.sub(r"\s+", " ", query).strip().lower()
    if not query_norm:
        return None
    channel = "voice" if "voice" in sys_txt.lower()[:2000] and "digital" not in query_norm else "digital"
    raw = json.dumps({"q": query_norm, "chunks": sorted(chunk_ids), "ch": channel})
    return hashlib.sha256(raw.encode()).hexdigest()


class _MemoryBackend:
    def __init__(self) -> None:
        self._d: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            expires, value = item
            if time.time() > expires:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._d[key] = (time.time() + settings.CACHE_TTL_S, value)
            self._d.move_to_end(key)
            while len(self._d) > settings.CACHE_MAX_ENTRIES:
                self._d.popitem(last=False)


class _RedisBackend:
    def __init__(self, url: str) -> None:
        import redis  # lazy import so the dep is optional
        self._r = redis.from_url(url, decode_responses=True)
        self._r.ping()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._r.get(f"gwcache:{key}")
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._r.setex(f"gwcache:{key}", settings.CACHE_TTL_S,
                      json.dumps(value, ensure_ascii=False))


_backend = None


def get_backend():
    global _backend
    if _backend is None:
        if settings.REDIS_URL:
            try:
                _backend = _RedisBackend(settings.REDIS_URL)
                log(logger, logging.INFO, "cache backend: redis")
            except Exception:
                log(logger, logging.ERROR,
                    "redis unreachable, falling back to in-memory cache")
                _backend = _MemoryBackend()
        else:
            _backend = _MemoryBackend()
            log(logger, logging.INFO, "cache backend: in-memory LRU",
                max_entries=settings.CACHE_MAX_ENTRIES, ttl_s=settings.CACHE_TTL_S)
    return _backend
