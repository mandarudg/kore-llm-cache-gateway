"""
Driver selection as code (agent-node route).

The XO agent prompt embeds UserQuery and a DriverData JSON blob; the LLM's
entire job is to filter drivers by OS compatibility and title keywords, then
echo the matching record. That is set filtering, not language understanding.

Strategy: attempt an unambiguous code match; on ANY doubt (0 matches, >1
equally-scored matches, unparseable data) return None and let the LLM decide.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.logging_setup import log

logger = logging.getLogger("driver_match")

_USER_QUERY_RE = re.compile(r"UserQuery:\s*(.+?)\s*-\s*DriverData:", re.DOTALL)
_DRIVER_DATA_RE = re.compile(r"DriverData:\s*(\{.*)", re.DOTALL)

_OS_PATTERNS = [
    (re.compile(r"windows\s*11", re.I), "windows 11"),
    (re.compile(r"windows\s*10\s*(64|32)?", re.I), "windows 10"),
    (re.compile(r"windows\s*8\.?1?", re.I), "windows 8"),
    (re.compile(r"windows\s*7", re.I), "windows 7"),
    (re.compile(r"mac\s*os|macos|os\s*x|sonoma|ventura|sequoia|monterey", re.I), "mac"),
    (re.compile(r"\blinux\b|\bubuntu\b", re.I), "linux"),
    (re.compile(r"chrome\s*os|chromebook", re.I), "chrome"),
]

# keyword groups scored against driver titles/descriptions
_KEYWORD_GROUPS = {
    "combo": ["combo package", "combo", "drivers and utilities"],
    "printer_driver": ["printer driver"],
    "scanner_driver": ["scanner driver", "epson scan"],
    "firmware": ["firmware"],
    "utility": ["utility", "software updater", "event manager", "scansmart",
                "ocr component", "netconfig", "remote print"],
}


def _extract(sys_txt: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    qm = _USER_QUERY_RE.search(sys_txt)
    dm = _DRIVER_DATA_RE.search(sys_txt)
    if not dm:
        return (qm.group(1).strip() if qm else None), None
    raw = dm.group(1)
    # DriverData is a JSON object followed by more prompt text -> balance braces
    depth, end = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return (qm.group(1).strip() if qm else None), None
    try:
        data = json.loads(raw[:end])
    except json.JSONDecodeError:
        return (qm.group(1).strip() if qm else None), None
    return (qm.group(1).strip() if qm else None), data


def _detect_os(query: str) -> Optional[str]:
    for pattern, canonical in _OS_PATTERNS:
        if pattern.search(query):
            return canonical
    return None


def _score(driver: Dict[str, Any], query_l: str) -> int:
    title_l = (driver.get("title") or "").lower()
    score = 0
    for _, phrases in _KEYWORD_GROUPS.items():
        for p in phrases:
            if p in query_l and p in title_l:
                score += 10 if len(p) > 6 else 5
    # generic token overlap tiebreaker
    for tok in set(re.findall(r"[a-z]{4,}", query_l)):
        if tok in title_l:
            score += 1
    return score


def try_match(sys_txt: str) -> Optional[Tuple[str, str]]:
    """Returns (response_content_json, match_note) or None to fall through."""
    query, data = _extract(sys_txt)
    if not query or not data or not isinstance(data.get("drivers"), list):
        return None
    query_l = query.lower()
    wanted_os = _detect_os(query_l)

    candidates: List[Dict[str, Any]] = data["drivers"]
    if wanted_os:
        candidates = [d for d in candidates
                      if wanted_os in (d.get("compatibleSystems") or "").lower()]
    if not candidates:
        return None

    # stable sort preserves DriverData array order among equals -- the platform
    # lists the most relevant entry first, and the LLM was observed to pick it
    scored = sorted(((_score(d, query_l), d) for d in candidates),
                    key=lambda x: -x[0])
    top_score, top = scored[0]
    if top_score < 10:
        return None  # no strong phrase-level match -> LLM decides
    if len(scored) > 1 and scored[1][0] == top_score:
        # tie: only resolve if the tied entries are near-duplicates of each
        # other (e.g. "...Installer Download" vs "...Installer"); otherwise
        # it's genuinely ambiguous -> LLM decides
        def _norm(d):
            t = (d.get("title") or "").lower()
            t = re.sub(r"\b(download|installer|v?\d+(\.\d+)*)\b", "", t)
            return re.sub(r"\s+", " ", t).strip()
        tied = [d for s, d in scored if s == top_score]
        if len({_norm(d) for d in tied}) > 1:
            return None

    content = json.dumps({
        "model": data.get("model", ""),
        "matchedDrivers": [{
            "isDriverAvailable": True,
            "title": top.get("title", ""),
            "downloadLink": top.get("downloadLink", ""),
            "description": top.get("description", ""),
        }],
    })
    note = f"os={wanted_os or 'any'} score={top_score} title={top.get('title', '')[:60]}"
    log(logger, logging.DEBUG, "driver code-match", note=note)
    return content, note
