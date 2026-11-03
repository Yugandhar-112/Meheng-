"""Safety layer: crisis detection (high recall) and objectionable-language flagging.

This is a deliberately simple keyword layer that runs BEFORE any model/LLM call so that
a person in distress always gets a fixed, safe reply. It is not a clinical tool.
"""
from __future__ import annotations

import re

try:
    from better_profanity import profanity
    profanity.load_censor_words()
except Exception:  # pragma: no cover
    profanity = None

_CRISIS = [
    r"\bkill\s+myself\b", r"\bend\s+(my\s+life|it\s+all)\b", r"\bsuicid(e|al)\b",
    r"\bwant(ed)?\s+to\s+die\b", r"\bdon'?t\s+want\s+to\s+(live|be\s+here)\b",
    r"\bno\s+reason\s+to\s+live\b", r"\b(hurt|harm)(ing)?\s+myself\b", r"\bself[- ]?harm\b",
    # Hindi / Hinglish
    r"\bkhudkushi\b", r"\bmarna\s+chah", r"\bjeena\s+nahi\b", r"\bjaan\s+de\s+d",
    r"आत्महत्या", r"मरना\s+चाह", r"जीना\s+नहीं",
]
_CRISIS_RE = re.compile("|".join(_CRISIS), re.IGNORECASE)

CRISIS_REPLY = {
    "en": (
        "I'm really sorry you're hurting this much, and I'm glad you told me. You deserve support "
        "from a real person right now.\n\n"
        "• In India you can call **Tele-MANAS at 14416** (24x7, free), or **112** in an emergency.\n"
        "• If you're elsewhere, please contact your local emergency number or a crisis line.\n"
        "• If you can, reach out to someone you trust and stay near them.\n\n"
        "I'm here to keep talking with you too. Would you like to tell me what's been happening?"
    ),
    "hi": (
        "Mujhe bahut afsos hai ki aap itna dard mehsoos kar rahe hain, aur mujhe khushi hai ki aapne bataya. "
        "Abhi kisi apne insaan se baat karna zaroori hai.\n\n"
        "• India mein **Tele-MANAS 14416** (24x7, muft) par call karein, ya emergency mein **112**.\n"
        "• Agar mumkin ho to kisi bharose ke insaan ke paas rahein.\n\n"
        "Main bhi yahan hoon. Kya aap bata sakte hain ki kya chal raha hai?"
    ),
}


def is_crisis(text: str) -> bool:
    return bool(_CRISIS_RE.search(text))


def is_objectionable(text: str) -> bool:
    return bool(profanity and profanity.contains_profanity(text))
