"""
Deterministic orchestrator rules.

These encode ONLY the unambiguous cases already spelled out in the XO
orchestrator prompt itself (Steps 1, 2 and small-talk fallback). Anything with
the slightest ambiguity returns None and falls through to the LLM. Precision
over recall: a rule that is right 99.9% of the time and covers 35% of turns
beats one that is right 97% of the time and covers 60%.

Every rule returns (verdict_json_string, rule_name) or None.
"""
import json
import re
from typing import Optional, Tuple

from core.schemas import has_active_dialog, looks_english

# ---------------------------------------------------------------- vocab

# Step-1 continue: bare confirmations / negations while a dialog is active.
# ("no" was observed in production logs to resolve to continue.)
_CONFIRM_NEGATE = {
    "yes", "yes please", "yeah", "yep", "yup", "sure", "ok", "okay", "k",
    "correct", "that's right", "thats right", "right", "sounds good",
    "go ahead", "continue", "proceed", "confirm", "confirmed",
    "no", "nope", "nah", "not yet", "no thanks", "no thank you",
}

# Standalone model number, e.g. "ET-2803", "et 2400", "WF-3820", "xp950",
# optionally prefixed with brand or short lead-ins the prompt calls out.
_MODEL_RE = re.compile(
    r"^(?:i\s+(?:have|am\s+using|need(?:\s+(?:it|support))?\s+for)\s+)?"
    r"(?:epson\s+)?[a-z]{1,4}[-\s]?\d{3,4}[a-z]{0,3}$",
    re.IGNORECASE,
)

# Step-2 hard overrides -> Default AgentTransfer
_AGENT_KEYWORDS = re.compile(
    r"\b(live\s*agent|human|representative|live\s*chat|talk\s+to\s+(?:someone|a\s+person)|"
    r"real\s+person|call\s+me|call\s*back|escalate|supervisor|manager|support\s+rep|"
    r"refund|money\s+back|return\s+label|\brma\b|send\s+it\s+back|"
    r"hard\s+copy\s+warranty|printed\s+copy|warranty\s+booklet|"
    r"(?:license|activation|product|serial)\s+key)\b",
    re.IGNORECASE,
)
# ...unless it's obviously a how-to question (FAQ-pattern override in the prompt)
_FAQ_PATTERN = re.compile(
    r"\b(how\s+(?:to|do\s+i|do\s+we|do\s+you|can\s+i|can\s+we)|"
    r"what\s+(?:are\s+the\s+steps|is\s+the\s+process)|where\s+can\s+i\s+(?:get|download|find))\b",
    re.IGNORECASE,
)
# "agent" alone is too ambiguous in printer land (user agent, agent software) -> require phrase
_BARE_AGENT_RE = re.compile(r"^\s*(agent|connect(\s+me)?(\s+again)?)\s*[.!]?\s*$", re.IGNORECASE)

# Small talk / thanks-only -> NoIntent_Indentified
_SMALL_TALK = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thankyou", "thx", "ty", "cool", "great", "nice",
    "how are you", "bye", "goodbye", "have a nice day",
}


# ---------------------------------------------------------------- verdict builders
# NB: intent JSON schema mirrors the orchestrator prompt exactly, including the
# platform's spelling "NoIntent_Indentified".

def _continue_verdict() -> str:
    return json.dumps({"language": "English", "category": "Category 2",
                       "fulfillment_type": "system_intent",
                       "winning_intents": ["continue"]})


def _agent_transfer_verdict() -> str:
    return json.dumps({"language": "English", "category": "Category 1",
                       "fulfillment_type": "single_intent",
                       "winning_intents": ["Default AgentTransfer"]})


def _no_intent_verdict() -> str:
    return json.dumps({"language": "English", "category": "Category 2",
                       "fulfillment_type": "NoIntent_Indentified",
                       "winning_intents": [""]})


# ---------------------------------------------------------------- engine

def evaluate(user_text: str, sys_txt: str, english_only: bool = True) -> Optional[Tuple[str, str]]:
    text = re.sub(r"\s+", " ", (user_text or "")).strip()
    if not text or len(text) > 120:
        # long inputs are never "bare confirmations"; let the LLM reason
        return None
    if english_only and not looks_english(text):
        return None

    lowered = text.lower().rstrip(".!?")

    # ---- Step-2 hard overrides (checked first, mirroring the prompt's order
    #      "return Default AgentTransfer and stop")
    if _BARE_AGENT_RE.match(text):
        return _agent_transfer_verdict(), "agent_transfer_bare"
    if _AGENT_KEYWORDS.search(text) and not _FAQ_PATTERN.search(text):
        return _agent_transfer_verdict(), "agent_transfer_keyword"

    # ---- Step-1 continue (requires an active dialog in Dialog Context)
    if has_active_dialog(sys_txt):
        if lowered in _CONFIRM_NEGATE:
            return _continue_verdict(), "continue_confirmation"
        if _MODEL_RE.match(lowered):
            return _continue_verdict(), "continue_model_number"

    # ---- Small talk (only when clearly nothing else)
    if lowered in _SMALL_TALK:
        return _no_intent_verdict(), "no_intent_small_talk"

    return None
