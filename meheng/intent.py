"""Intent matching for built-in skills (time, date, jokes ...).

Hybrid: multilingual Sentence-BERT embeddings + cosine similarity propose an intent, and a small
keyword check confirms it (SBERT alone matches short romanised Hinglish too loosely).
Fallback: difflib string similarity. Anything unconfirmed -> "none" and goes to the LLM.
"""
from __future__ import annotations

import difflib
import re

INTENTS: dict[str, list[str]] = {
    "greeting": ["hi", "hello", "hey there", "good morning", "good evening", "namaste",
                 "hello meheng", "kaise ho", "hey"],
    "goodbye": ["bye", "goodbye", "see you later", "talk to you later", "good night",
                "alvida", "chalo bye"],
    "time": ["what time is it", "tell me the time", "current time", "what's the time",
             "time batao", "kitne baje hain", "samay kya hua hai"],
    "date": ["what is today's date", "what day is it today", "tell me the date",
             "aaj ki date kya hai", "aaj kaun sa din hai", "today's date"],
    "joke": ["tell me a joke", "make me laugh", "say something funny", "joke sunao",
             "koi joke sunao", "i want to hear a joke"],
    "help": ["what can you do", "help me", "how can you help", "your features",
             "tum kya kar sakte ho"],
    "identity": ["who are you", "what is your name", "tum kaun ho", "tell me about yourself"],
}

# An intent from SBERT/difflib is only accepted if one of these keywords is in the text.
REQUIRED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "greeting": ("hi", "hello", "hey", "namaste", "good morning", "good evening", "kaise ho"),
    "goodbye": ("bye", "goodbye", "see you", "talk to you", "good night", "alvida"),
    "time": ("time", "baje", "samay", "clock"),
    "date": ("date", "day", "din", "tarikh", "taarikh", "tareekh"),
    "joke": ("joke", "funny", "laugh", "hasa", "mazak", "chutkula"),
    "help": ("help", "can you do", "kar sakte", "features", "madad"),
    "identity": ("who", "name", "kaun", "naam", "yourself"),
}

MAX_TOKENS = 6          # long messages are never treated as simple commands
SBERT_THRESHOLD = 0.65
DIFFLIB_THRESHOLD = 0.75


def _has_keyword(intent: str, tokens: list[str], text: str) -> bool:
    keys = REQUIRED_KEYWORDS.get(intent)
    if not keys:
        return True
    for k in keys:
        if " " in k:
            if k in text:
                return True
        elif any(t == k or (len(k) > 3 and t.startswith(k)) for t in tokens):
            return True
    return False


class IntentMatcher:
    def __init__(self, use_embeddings: bool = True,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.examples = [(i, p) for i, ps in INTENTS.items() for p in ps]
        self.mode = "difflib"
        self._model = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self._emb = self._model.encode([p for _, p in self.examples], normalize_embeddings=True)
                self.mode = "sbert"
            except Exception as e:
                print(f"[meheng] SBERT intent matcher unavailable ({type(e).__name__}); using difflib")

    def match(self, text: str) -> tuple[str, float]:
        t = re.sub(r"[^\w\s']", " ", text.lower()).strip()
        toks = t.split()
        if not t or len(toks) > MAX_TOKENS:
            return "none", 0.0
        if self._model is not None:
            q = self._model.encode([t], normalize_embeddings=True)[0]
            sims = self._emb @ q
            k = int(sims.argmax())
            score, intent = float(sims[k]), self.examples[k][0]
            if score >= SBERT_THRESHOLD and _has_keyword(intent, toks, t):
                return intent, score
            return "none", score
        best, best_i = 0.0, "none"
        for intent, p in self.examples:
            s = difflib.SequenceMatcher(None, t, p).ratio()
            if len(p.split()) >= 2 and p in t:
                s = max(s, 0.9)
            if len(p.split()) == 1 and toks[:1] == [p] and len(toks) <= 2:
                s = max(s, 0.85)
            if s > best:
                best, best_i = s, intent
        if best >= DIFFLIB_THRESHOLD and _has_keyword(best_i, toks, t):
            return best_i, best
        return "none", best