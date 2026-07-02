"""
Central configuration. Every knob is an environment variable so behavior can be
changed on Railway without a redeploy of code (just restart with new vars).
"""
import os


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    # ------------------------------------------------------------------ upstream
    # OpenAI by default. For Azure OpenAI set:
    #   UPSTREAM_BASE_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>
    #   UPSTREAM_AUTH_STYLE=azure   (sends api-key header + api-version query param)
    UPSTREAM_BASE_URL: str = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    UPSTREAM_API_KEY: str = os.getenv("UPSTREAM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    UPSTREAM_AUTH_STYLE: str = os.getenv("UPSTREAM_AUTH_STYLE", "openai")  # openai | azure
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-10-21")
    UPSTREAM_TIMEOUT_S: float = _float("UPSTREAM_TIMEOUT_S", 60.0)

    # Optional per-route model override ("" = keep whatever XO sent).
    # Use these to silently downgrade a route, e.g. SEARCH_MODEL_OVERRIDE=gpt-4o-mini
    ORCH_MODEL_OVERRIDE: str = os.getenv("ORCH_MODEL_OVERRIDE", "")
    AGENT_MODEL_OVERRIDE: str = os.getenv("AGENT_MODEL_OVERRIDE", "")
    SEARCH_MODEL_OVERRIDE: str = os.getenv("SEARCH_MODEL_OVERRIDE", "")

    # ------------------------------------------------------------------ gateway auth
    # If set, XO must send this as Authorization: Bearer <token> (configure as the
    # Custom LLM "API key" in XO). Empty = no gateway auth (not recommended).
    GATEWAY_API_KEY: str = os.getenv("GATEWAY_API_KEY", "")

    # ------------------------------------------------------------------ feature flags
    # SHADOW_MODE: rules/cache still execute + log their verdict, but every request
    # is served by the upstream LLM. Use this for the first 1-2 weeks to measure
    # agreement risk-free, then flip to false.
    SHADOW_MODE: bool = _bool("SHADOW_MODE", True)

    ORCH_RULES_ENABLED: bool = _bool("ORCH_RULES_ENABLED", True)
    AGENT_CODE_MATCH_ENABLED: bool = _bool("AGENT_CODE_MATCH_ENABLED", True)
    SEARCH_CACHE_ENABLED: bool = _bool("SEARCH_CACHE_ENABLED", True)

    # Restructure orchestrator prompt (static prefix first, dynamic tail last) so
    # OpenAI prefix caching kicks in. Ships OFF; enable after shadow comparison.
    PREFIX_RESTRUCTURE_ENABLED: bool = _bool("PREFIX_RESTRUCTURE_ENABLED", False)

    # Rules only fire on inputs that look like English; everything else goes to
    # the LLM untouched. Flip off later when LATAM rules are added.
    RULES_ENGLISH_ONLY: bool = _bool("RULES_ENGLISH_ONLY", True)

    # ------------------------------------------------------------------ cache
    # In-memory by default; set REDIS_URL for persistence across restarts/replicas.
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    CACHE_TTL_S: int = _int("CACHE_TTL_S", 6 * 3600)          # 6h default
    CACHE_MAX_ENTRIES: int = _int("CACHE_MAX_ENTRIES", 5000)  # in-memory LRU bound

    # ------------------------------------------------------------------ traffic bank
    # On Railway, attach a Volume and mount it at /data so JSONL survives deploys.
    TRAFFIC_BANK_DIR: str = os.getenv("TRAFFIC_BANK_DIR", "/data/traffic")
    TRAFFIC_BANK_ENABLED: bool = _bool("TRAFFIC_BANK_ENABLED", True)
    # full = entire request/response bodies (needed for fine-tuning exports)
    # slim = user input + verdicts only (smaller disk footprint)
    TRAFFIC_BANK_MODE: str = os.getenv("TRAFFIC_BANK_MODE", "full")
    TRAFFIC_BANK_MAX_MB_PER_FILE: int = _int("TRAFFIC_BANK_MAX_MB_PER_FILE", 100)

    # ------------------------------------------------------------------ logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_BODIES: bool = _bool("LOG_BODIES", False)  # log full payloads to stdout (verbose!)

    # ------------------------------------------------------------------ misc
    PORT: int = _int("PORT", 8000)  # Railway injects PORT automatically


settings = Settings()
