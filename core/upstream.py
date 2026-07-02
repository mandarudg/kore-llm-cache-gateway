"""
Upstream LLM client (OpenAI or Azure OpenAI) + optional prompt restructuring.

Restructuring: XO injects dynamic context (history, chunks, dialog state) into
the MIDDLE of the orchestrator system prompt, which defeats OpenAI's automatic
prefix caching (we observed cached_tokens: 0 on 3.7K-token prompts). When
PREFIX_RESTRUCTURE_ENABLED=true the gateway splits the system prompt at the
known 'Context:' boundary, keeps the ~3K-token static instruction block as a
stable system prefix, and moves the dynamic tail into the user message. Same
information, same order of reading for the model, but now the prefix is
byte-identical across turns -> 50% cached-input pricing + lower TTFT.
"""
import copy
import logging
import re
import time
from typing import Any, Dict, Tuple

import httpx

from core.config import settings
from core.logging_setup import log

logger = logging.getLogger("upstream")

# The observed orchestrator template is [static A][dynamic middle][static B]:
#   static A: role + schema instructions            (~260 tokens)
#   dynamic : Context 1/2/3 + chunk inputs           (changes every turn)
#   static B: "Decision order (top to bottom)" steps (~2.5-3K tokens)
# We rebuild as system=[A+B] (stable, cacheable prefix) and prepend the dynamic
# middle to the user message.
_DYN_START_RE = re.compile(r"(Context:\s*1\.\s*Conversation History:)", re.IGNORECASE)
_DYN_END_RE = re.compile(r"(Decision order\s*\(top to bottom\)\s*:)", re.IGNORECASE)

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT_S)
    return _client


def _url_and_headers() -> Tuple[str, Dict[str, str]]:
    if settings.UPSTREAM_AUTH_STYLE == "azure":
        url = f"{settings.UPSTREAM_BASE_URL.rstrip('/')}/chat/completions?api-version={settings.AZURE_API_VERSION}"
        headers = {"api-key": settings.UPSTREAM_API_KEY}
    else:
        url = f"{settings.UPSTREAM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.UPSTREAM_API_KEY}"}
    headers["Content-Type"] = "application/json"
    return url, headers


def restructure_for_prefix_cache(body: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Split orchestrator system prompt into (static system prefix, dynamic user tail).
    Returns (new_body, changed). Never mutates the original body."""
    try:
        msgs = body.get("messages", [])
        sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
        if sys_idx is None:
            return body, False
        content = msgs[sys_idx].get("content")
        # normalize list-style content to plain text
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "text")
        if not isinstance(content, str):
            return body, False
        m_start = _DYN_START_RE.search(content)
        if not m_start:
            return body, False  # template changed or not the orchestrator -> leave alone
        m_end = _DYN_END_RE.search(content, m_start.end())
        if m_end:
            # three-part template: prefix = A + B, dynamic = the middle
            static_part = (content[: m_start.start()].rstrip()
                           + "\n\n" + content[m_end.start():].strip())
            dynamic_part = content[m_start.start(): m_end.start()].strip()
        else:
            # unknown tail -> conservative two-part split
            static_part = content[: m_start.start()].rstrip()
            dynamic_part = content[m_start.start():]

        new_body = copy.deepcopy(body)
        new_msgs = new_body["messages"]
        new_msgs[sys_idx]["content"] = static_part
        # prepend the dynamic block to the (last) user message so the model still
        # reads instructions -> context -> user input in the same order
        for i in range(len(new_msgs) - 1, -1, -1):
            if new_msgs[i].get("role") == "user":
                uc = new_msgs[i].get("content")
                if isinstance(uc, list):
                    uc = " ".join(p.get("text", "") for p in uc
                                  if isinstance(p, dict) and p.get("type") == "text")
                new_msgs[i]["content"] = f"{dynamic_part}\n\nUser input: {uc}"
                return new_body, True
        return body, False
    except Exception:
        log(logger, logging.WARNING, "prefix restructure failed, passing through")
        return body, False


def clean_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Strip XO bookkeeping fields OpenAI would reject, and known junk tokens."""
    b = {k: v for k, v in body.items() if k not in ("startedAt", "respondedAt")}
    return b


async def chat_completion(body: Dict[str, Any], route: str,
                          model_override: str = "") -> Tuple[Dict[str, Any], int]:
    """Forward to upstream; returns (response_json, latency_ms)."""
    b = clean_body(body)
    if model_override:
        b["model"] = model_override
    url, headers = _url_and_headers()
    t0 = time.perf_counter()
    resp = await get_client().post(url, json=b, headers=headers)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if resp.status_code != 200:
        log(logger, logging.ERROR, "upstream error", route=route,
            status=resp.status_code, body=resp.text[:500], latency_ms=latency_ms)
        resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    log(logger, logging.INFO, "upstream ok", route=route, model=data.get("model"),
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        completion_tokens=usage.get("completion_tokens"))
    return data, latency_ms
