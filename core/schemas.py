"""
Helpers to (a) pull the pieces we need out of XO's giant single-prompt payloads
and (b) fabricate OpenAI-schema responses for rule/cache hits so XO cannot tell
the difference between the gateway and a real model.
"""
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- request parsing
# XO speaks two OpenAI dialects depending on the model configured:
#   Chat Completions: {"model", "messages": [...]}          -> /v1/chat/completions
#   Responses API:    {"model", "input": [...], "text":{}}  -> /v1/responses
# All helpers below accept either shape.

def is_responses_request(body: Dict[str, Any]) -> bool:
    return "input" in body and "messages" not in body


def get_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = body.get("messages")
    if msgs is None:
        msgs = body.get("input")
    if isinstance(msgs, str):  # Responses API also allows a bare string input
        return [{"role": "user", "content": msgs}]
    return msgs or []


def _content_to_text(content: Any) -> str:
    """XO sometimes sends content as a string, sometimes as [{'type':'text','text':...}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def system_text(body: Dict[str, Any]) -> str:
    for m in get_messages(body):
        if m.get("role") == "system":
            return _content_to_text(m.get("content"))
    return ""


def last_user_text(body: Dict[str, Any]) -> str:
    for m in reversed(get_messages(body)):
        if m.get("role") == "user":
            return _content_to_text(m.get("content"))
    return ""


_ASCII_RE = re.compile(r"^[\x00-\x7F]+$")


def looks_english(text: str) -> bool:
    """Cheap heuristic: pure-ASCII and no obvious Spanish/Portuguese markers.
    Deliberately conservative -- when unsure we say 'not English' so the LLM handles it."""
    t = text.strip().lower()
    if not t or not _ASCII_RE.match(t):
        return False
    non_en_markers = (" el ", " la ", " los ", " una ", " não ", " nao ", " por favor",
                      "hola", "gracias", "obrigad", " si ", " sí ", "necesito", "preciso",
                      "quiero", "ayuda", "ajuda", "impresora", "impressora")
    padded = f" {t} "
    return not any(m in padded for m in non_en_markers)


def extract_dialog_context(sys_txt: str) -> Optional[Dict[str, Any]]:
    """Pull the `Dialog Context: {...}` JSON object out of the orchestrator system prompt."""
    m = re.search(r"Dialog Context:\s*(\{.*?\})\s*3\.\s*User Context", sys_txt, re.DOTALL)
    if not m:
        m = re.search(r"Dialog Context:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", sys_txt)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def has_active_dialog(sys_txt: str) -> bool:
    ctx = extract_dialog_context(sys_txt)
    if not ctx:
        return False
    return bool(ctx.get("dialog_name")) and bool(ctx.get("current_node"))


# --------------------------------------------------------------------------- response fabrication

def openai_response(content: str, model: str, prompt_tokens: int = 0,
                    completion_tokens: int = 0, gateway_layer: str = "rules") -> Dict[str, Any]:
    """A chat.completion body indistinguishable (schema-wise) from OpenAI's."""
    return {
        "id": f"chatcmpl-gw-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "refusal": None,
                        "annotations": []},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0,
                                          "accepted_prediction_tokens": 0,
                                          "rejected_prediction_tokens": 0},
        },
        "service_tier": "default",
        "system_fingerprint": f"gw_{gateway_layer}",
    }


def responses_response(content: str, model: str, prompt_tokens: int = 0,
                       completion_tokens: int = 0, gateway_layer: str = "rules") -> Dict[str, Any]:
    """A Responses-API body indistinguishable (schema-wise) from OpenAI's."""
    return {
        "id": f"resp_gw{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": model,
        "output": [{
            "id": f"msg_gw{uuid.uuid4().hex[:24]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }],
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "metadata": {"gateway_layer": gateway_layer},
    }


def fabricate_response(request_body: Dict[str, Any], content: str, model: str,
                       prompt_tokens: int, completion_tokens: int,
                       gateway_layer: str) -> Dict[str, Any]:
    """Build a local response in the SAME dialect the request arrived in."""
    if is_responses_request(request_body):
        return responses_response(content, model, prompt_tokens,
                                  completion_tokens, gateway_layer)
    return openai_response(content, model, prompt_tokens,
                           completion_tokens, gateway_layer)


def assistant_text(response: Dict[str, Any]) -> str:
    """Extract assistant text from either dialect's response."""
    # Chat Completions
    choices = response.get("choices")
    if choices:
        return (choices[0].get("message") or {}).get("content") or ""
    # Responses API
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    return part.get("text") or ""
    return ""


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)
