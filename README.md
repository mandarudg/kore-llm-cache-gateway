# Kore.ai XO Smart Inference Gateway

OpenAI-compatible proxy that sits between Kore.ai XO's Custom LLM feature and
OpenAI/Azure OpenAI. Serves what it can locally (rules, deterministic code,
cache), forwards the rest, and logs every exchange to a traffic bank for
fine-tuning exports.

```
XO feature ──> gateway route ──> [rules / code / cache] ──hit──> instant response
                                        │ miss
                                        └──> OpenAI / Azure ──> response
                                                   │
                                             traffic bank (JSONL)
```

## Routes -> XO Custom LLM configuration

| XO feature | Endpoint to configure in XO |
|---|---|
| Intent detection / orchestration | `https://<app>.up.railway.app/orchestrator/v1/chat/completions` |
| Driver-selection agent node | `https://<app>.up.railway.app/agent-node/v1/chat/completions` |
| Answer generation / RAG search | `https://<app>.up.railway.app/search/v1/chat/completions` |
| Anything else (safe default) | `https://<app>.up.railway.app/passthrough/v1/chat/completions` |

In XO, set the Custom LLM **API key** to the value of `GATEWAY_API_KEY` (sent
as `Authorization: Bearer ...`). Both `/route/v1/chat/completions` and
`/route/chat/completions` paths are accepted.

Ops endpoints: `GET /healthz` (liveness), `GET /metrics` (per-route/per-layer
counters, offload %, shadow agreement %).

## Deploy on Railway

1. Push this folder to a GitHub repo, create a Railway service from it
   (Dockerfile is auto-detected; `railway.json` sets `/healthz` as healthcheck).
2. **Attach a Volume** to the service, mount path `/data`. Without it the
   traffic bank is wiped on every deploy.
3. Set variables (minimum):
   ```
   UPSTREAM_API_KEY = sk-...            # your OpenAI key
   GATEWAY_API_KEY  = <generate a long random string>
   ```
4. Deploy. Check logs for the `gateway starting` line, then hit `/healthz`.
5. Point ONE low-risk XO feature at `/passthrough` first to validate the
   XO <-> gateway <-> OpenAI plumbing, then move the three real features over.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `UPSTREAM_API_KEY` (or `OPENAI_API_KEY`) | — | OpenAI/Azure key. **Required.** |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | For Azure: `https://<res>.openai.azure.com/openai/deployments/<dep>` |
| `UPSTREAM_AUTH_STYLE` | `openai` | `azure` sends `api-key` header + `api-version` param |
| `AZURE_API_VERSION` | `2024-10-21` | Azure only |
| `UPSTREAM_TIMEOUT_S` | `60` | Upstream request timeout |
| `GATEWAY_API_KEY` | *(empty)* | Bearer token XO must send. Empty = open endpoint (don't). |
| `SHADOW_MODE` | `true` | Rules/cache compute + log verdicts but LLM serves everything. **Ships ON.** |
| `ORCH_RULES_ENABLED` | `true` | Orchestrator rule shortcuts |
| `AGENT_CODE_MATCH_ENABLED` | `true` | Driver selection as code |
| `SEARCH_CACHE_ENABLED` | `true` | Exact-match answer cache |
| `PREFIX_RESTRUCTURE_ENABLED` | `false` | Rebuild orchestrator prompt for OpenAI prefix caching |
| `RULES_ENGLISH_ONLY` | `true` | Rules skip non-English inputs (LLM handles them) |
| `ORCH_MODEL_OVERRIDE` | *(empty)* | e.g. your fine-tuned model id `ft:gpt-4o-mini:...` |
| `AGENT_MODEL_OVERRIDE` | *(empty)* | e.g. `gpt-4o-mini` |
| `SEARCH_MODEL_OVERRIDE` | *(empty)* | e.g. `gpt-4o-mini` for the A/B downgrade test |
| `REDIS_URL` | *(empty)* | Optional Redis for cache persistence (Railway Redis plugin works) |
| `CACHE_TTL_S` | `21600` | Cache entry lifetime (6h) |
| `CACHE_MAX_ENTRIES` | `5000` | In-memory LRU bound |
| `TRAFFIC_BANK_DIR` | `/data/traffic` | Where JSONL lands (mount a Volume here) |
| `TRAFFIC_BANK_ENABLED` | `true` | |
| `TRAFFIC_BANK_MODE` | `full` | `full` = keep bodies (needed for fine-tuning); `slim` = verdicts only |
| `TRAFFIC_BANK_MAX_MB_PER_FILE` | `100` | Size rotation per daily file |
| `LOG_LEVEL` | `INFO` | `DEBUG` for deep dives |
| `LOG_BODIES` | `false` | Log full payloads to stdout (very verbose) |

## Logging

Every log line is one JSON object on stdout (Railway log explorer friendly).
Key lines per request:

```json
{"msg":"request in","route":"orchestrator","request_id":"a1b2...","user_input":"no","shadow":true}
{"msg":"shadow comparison","route":"orchestrator","rule":"continue_confirmation","agree":true}
{"msg":"upstream ok","route":"orchestrator","latency_ms":702,"prompt_tokens":3763,"cached_tokens":0}
{"msg":"request out","route":"orchestrator","layer":"llm","latency_ms":714}
```

Filter examples in Railway: `layer:"rules"` (local serves), `shadow_agree:false`
(rule disagreements to review), `msg:"upstream error"`.

## Operating playbook

**Week 1-2 (shadow):** everything ships with `SHADOW_MODE=true`. XO users see
zero change; the gateway measures. Watch `GET /metrics` ->
`shadow_agreement_pct` per route.

**Go-live criteria:** flip `SHADOW_MODE=false` when orchestrator shadow
agreement is >= 98% over >= 2,000 turns. Review every `shadow_agree:false`
record in the traffic bank first — most disagreements are rule bugs you can
fix or vocabulary you should remove.

**Enable prefix caching:** set `PREFIX_RESTRUCTURE_ENABLED=true` after a
shadow-style spot check (run 50 traffic-bank requests through both prompt
shapes offline and diff the outputs). Expect `cached_tokens` in the
`upstream ok` log lines to jump from 0 to ~2,500 on the second-and-later
orchestrator turns.

**Fine-tune (monthly):**
```bash
# on the Railway service shell, or download the volume contents first
python scripts/export_finetune_jsonl.py --route orchestrator --strip-static
# upload the file at platform.openai.com/finetune, base model gpt-4o-mini
# when done, set: ORCH_MODEL_OVERRIDE=ft:gpt-4o-mini-2024-07-18:org:name:id
```
If you train with `--strip-static`, the served prompt must also be stripped —
that mode requires `PREFIX_RESTRUCTURE_ENABLED=true` plus a small change to
drop the static system part for the fine-tuned model (Phase 3; ask before
enabling). Training WITHOUT `--strip-static` works with zero serving changes
and is the safer first fine-tune.

**Rollback:** any route can be reverted instantly by repointing the XO feature
back to the original OpenAI config, or by setting that route's flag to false.
No code changes, no redeploys.

## Safety properties

- Unknown/changed prompt templates never match rule/code/cache extractors ->
  automatic LLM passthrough.
- Any handler exception triggers a raw passthrough retry before failing.
- Rule verdicts replicate the platform's exact JSON schema (including the
  platform's spelling `NoIntent_Indentified`).
- Cache keys include chunk IDs -> KB re-index invalidates stale answers.
- Fabricated responses match the OpenAI schema byte-for-byte, including the
  `usage` block XO parses; `system_fingerprint` is prefixed `gw_` so gateway
  serves are identifiable in XO-side logs too.
