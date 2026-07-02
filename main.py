"""
Kore.ai XO Smart Inference Gateway
==================================
OpenAI-compatible proxy. Configure each XO feature's Custom LLM endpoint to one
of the three routes; the gateway serves what it can locally and forwards the
rest to OpenAI/Azure, logging everything to the traffic bank for fine-tuning.

Routes (XO Custom LLM endpoint URLs):
  POST /orchestrator/v1/chat/completions   intent detection (rules -> LLM)
  POST /agent-node/v1/chat/completions     driver selection (code -> LLM)
  POST /search/v1/chat/completions         RAG answer gen (cache -> LLM)
  POST /passthrough/v1/chat/completions    verbatim forward (safe default)
  GET  /healthz                            liveness
  GET  /metrics                            per-route/per-layer counters
"""
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from core import cache as cache_mod
from core import driver_match, rules, traffic_bank, upstream
from core.config import settings
from core.logging_setup import log, setup_logging
from core.schemas import est_tokens, last_user_text, openai_response, system_text

setup_logging()
logger = logging.getLogger("gateway")
app = FastAPI(title="Kore XO Smart Inference Gateway", docs_url=None, redoc_url=None)

START_TIME = time.time()
METRICS: Dict[str, Dict[str, int]] = {}  # {route: {layer: count}}


def _bump(route: str, layer: str) -> None:
    METRICS.setdefault(route, {}).setdefault(layer, 0)
    METRICS[route][layer] += 1


def _check_auth(request: Request) -> None:
    if not settings.GATEWAY_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {settings.GATEWAY_API_KEY}":
        raise HTTPException(status_code=401, detail="invalid gateway api key")


async def _read_body(request: Request) -> Dict[str, Any]:
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")


def _bank(route: str, rid: str, layer: str, body: Dict[str, Any],
          response: Dict[str, Any], latency_ms: int, **extra) -> None:
    entry = {
        "request_id": rid, "route": route, "layer": layer,
        "latency_ms": latency_ms,
        "user_input": last_user_text(body)[:1000],
        "model": response.get("model"),
        "usage": response.get("usage"),
        "assistant_content": (
            (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        )[:6000],
        "request_body": body,
        "response_body": response,
        **extra,
    }
    traffic_bank.record(route, entry)


# ============================================================== route handlers

async def _serve_orchestrator(body: Dict[str, Any], rid: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Returns (response, layer, extra_log_fields)."""
    sys_txt = system_text(body)
    user_txt = last_user_text(body)
    verdict: Optional[Tuple[str, str]] = None

    if settings.ORCH_RULES_ENABLED:
        verdict = rules.evaluate(user_txt, sys_txt, settings.RULES_ENGLISH_ONLY)

    if verdict and not settings.SHADOW_MODE:
        content, rule_name = verdict
        resp = openai_response(content, model="gateway-rules",
                               prompt_tokens=est_tokens(sys_txt),
                               completion_tokens=est_tokens(content),
                               gateway_layer="rules")
        return resp, "rules", {"rule": rule_name}

    # LLM path (optionally restructured for prefix caching)
    fwd = body
    restructured = False
    if settings.PREFIX_RESTRUCTURE_ENABLED:
        fwd, restructured = upstream.restructure_for_prefix_cache(body)
    resp, _ = await upstream.chat_completion(fwd, "orchestrator",
                                             settings.ORCH_MODEL_OVERRIDE)
    extra: Dict[str, Any] = {"prefix_restructured": restructured}
    if verdict:  # shadow mode: compare rule verdict vs LLM ground truth
        llm_content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        agree = _verdicts_agree(verdict[0], llm_content)
        extra.update({"shadow_rule": verdict[1], "shadow_rule_verdict": verdict[0],
                      "shadow_agree": agree})
        log(logger, logging.INFO, "shadow comparison", route="orchestrator",
            request_id=rid, rule=verdict[1], agree=agree)
        _bump("orchestrator", "shadow_agree" if agree else "shadow_disagree")
    return resp, "llm", extra


def _verdicts_agree(rule_json: str, llm_content: str) -> bool:
    """Loose agreement: same fulfillment_type and same winning intent set."""
    import json as _json
    try:
        a = _json.loads(rule_json)
        b = _json.loads(llm_content)
        return (a.get("fulfillment_type") == b.get("fulfillment_type")
                and a.get("winning_intents") == b.get("winning_intents"))
    except Exception:
        return False


async def _serve_agent_node(body: Dict[str, Any], rid: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    sys_txt = system_text(body)
    match: Optional[Tuple[str, str]] = None
    if settings.AGENT_CODE_MATCH_ENABLED:
        match = driver_match.try_match(sys_txt)

    if match and not settings.SHADOW_MODE:
        content, note = match
        resp = openai_response(content, model="gateway-code",
                               prompt_tokens=est_tokens(sys_txt),
                               completion_tokens=est_tokens(content),
                               gateway_layer="code_match")
        return resp, "code_match", {"match_note": note}

    resp, _ = await upstream.chat_completion(body, "agent-node",
                                             settings.AGENT_MODEL_OVERRIDE)
    extra: Dict[str, Any] = {}
    if match:
        llm_content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        # agreement = same downloadLink chosen
        agree = _same_download_link(match[0], llm_content)
        extra.update({"shadow_match_note": match[1], "shadow_agree": agree})
        _bump("agent-node", "shadow_agree" if agree else "shadow_disagree")
    return resp, "llm", extra


def _same_download_link(code_json: str, llm_content: str) -> bool:
    import json as _json
    import re as _re
    try:
        code_link = _json.loads(code_json)["matchedDrivers"][0]["downloadLink"]
        llm_links = _re.findall(r"https://\S+?\.exe", llm_content)
        return code_link in llm_links
    except Exception:
        return False


async def _serve_search(body: Dict[str, Any], rid: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    sys_txt = system_text(body)
    user_txt = last_user_text(body)
    key = cache_mod.derive_key(user_txt, sys_txt) if settings.SEARCH_CACHE_ENABLED else None

    if key:
        cached = cache_mod.get_backend().get(key)
        if cached and not settings.SHADOW_MODE:
            cached["id"] = f"chatcmpl-gw-{uuid.uuid4().hex[:24]}"  # fresh id per serve
            cached["system_fingerprint"] = "gw_cache"
            return cached, "cache", {"cache_key": key[:16]}

    resp, _ = await upstream.chat_completion(body, "search",
                                             settings.SEARCH_MODEL_OVERRIDE)
    if key:
        try:
            cache_mod.get_backend().set(key, resp)
        except Exception:
            log(logger, logging.WARNING, "cache set failed", request_id=rid)
    return resp, "llm", {"cache_key": (key or "")[:16], "cache_store": bool(key)}


# ============================================================== generic wrapper

async def _handle(route: str, request: Request,
                  handler: Callable) -> JSONResponse:
    _check_auth(request)
    rid = uuid.uuid4().hex[:12]
    body = await _read_body(request)
    t0 = time.perf_counter()
    log(logger, logging.INFO, "request in", route=route, request_id=rid,
        model=body.get("model"), user_input=last_user_text(body)[:200],
        shadow=settings.SHADOW_MODE)
    if settings.LOG_BODIES:
        log(logger, logging.DEBUG, "request body", route=route, request_id=rid, body=body)

    try:
        resp, layer, extra = await handler(body, rid)
    except HTTPException:
        raise
    except Exception as exc:
        # never leave XO hanging: last-ditch verbatim passthrough
        log(logger, logging.ERROR, "handler crashed, attempting raw passthrough",
            route=route, request_id=rid, error=str(exc))
        _bump(route, "error")
        try:
            resp, _ = await upstream.chat_completion(body, route)
            layer, extra = "llm_recovery", {"recovered_from": str(exc)[:200]}
        except Exception as exc2:
            log(logger, logging.CRITICAL, "passthrough also failed",
                route=route, request_id=rid, error=str(exc2))
            raise HTTPException(status_code=502, detail="upstream failure")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    _bump(route, layer)
    log(logger, logging.INFO, "request out", route=route, request_id=rid,
        layer=layer, latency_ms=latency_ms, **{k: v for k, v in extra.items()
                                               if k not in ("shadow_rule_verdict",)})
    _bank(route, rid, layer, body, resp, latency_ms, **extra)
    return JSONResponse(resp)


# ============================================================== endpoints
# Both /route/v1/chat/completions and /route/chat/completions are accepted --
# XO builds paths slightly differently across versions.

@app.post("/orchestrator/v1/chat/completions")
@app.post("/orchestrator/chat/completions")
async def orchestrator(request: Request):
    return await _handle("orchestrator", request, _serve_orchestrator)


@app.post("/agent-node/v1/chat/completions")
@app.post("/agent-node/chat/completions")
async def agent_node(request: Request):
    return await _handle("agent-node", request, _serve_agent_node)


@app.post("/search/v1/chat/completions")
@app.post("/search/chat/completions")
async def search(request: Request):
    return await _handle("search", request, _serve_search)


async def _passthrough_handler(body: Dict[str, Any], rid: str):
    resp, _ = await upstream.chat_completion(body, "passthrough")
    return resp, "llm", {}


@app.post("/passthrough/v1/chat/completions")
@app.post("/passthrough/chat/completions")
async def passthrough(request: Request):
    return await _handle("passthrough", request, _passthrough_handler)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "uptime_s": int(time.time() - START_TIME),
            "shadow_mode": settings.SHADOW_MODE}


@app.get("/metrics")
async def metrics():
    """Per-route/per-layer counters -- the containment story in one JSON blob."""
    out = {"uptime_s": int(time.time() - START_TIME),
           "shadow_mode": settings.SHADOW_MODE, "routes": {}}
    for route, layers in METRICS.items():
        total = sum(v for k, v in layers.items()
                    if k in ("rules", "code_match", "cache", "llm", "llm_recovery"))
        served_local = sum(layers.get(k, 0) for k in ("rules", "code_match", "cache"))
        route_view = dict(layers)
        route_view["total_served"] = total
        route_view["offload_pct"] = round(100 * served_local / total, 1) if total else 0.0
        agree = layers.get("shadow_agree", 0)
        disagree = layers.get("shadow_disagree", 0)
        if agree + disagree:
            route_view["shadow_agreement_pct"] = round(100 * agree / (agree + disagree), 1)
        out["routes"][route] = route_view
    return out


@app.on_event("startup")
async def _startup():
    log(logger, logging.INFO, "gateway starting",
        shadow_mode=settings.SHADOW_MODE,
        upstream=settings.UPSTREAM_BASE_URL,
        auth_style=settings.UPSTREAM_AUTH_STYLE,
        orch_rules=settings.ORCH_RULES_ENABLED,
        agent_code_match=settings.AGENT_CODE_MATCH_ENABLED,
        search_cache=settings.SEARCH_CACHE_ENABLED,
        prefix_restructure=settings.PREFIX_RESTRUCTURE_ENABLED,
        traffic_bank=settings.TRAFFIC_BANK_DIR if settings.TRAFFIC_BANK_ENABLED else "disabled")
    if not settings.UPSTREAM_API_KEY:
        log(logger, logging.CRITICAL, "UPSTREAM_API_KEY / OPENAI_API_KEY is not set")
    if not settings.GATEWAY_API_KEY:
        log(logger, logging.WARNING,
            "GATEWAY_API_KEY not set -- endpoint is unauthenticated")
