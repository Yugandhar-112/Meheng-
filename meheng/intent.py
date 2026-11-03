"""Intent matching for built-in skills (time, date, jokes ...).

Primary: multilingual Sentence-BERT embeddings + cosine similarity to example phrases
(handles paraphrases and Hindi/Hinglish). Fallback: difflib string similarity.
Anything below threshold -> "none" and goes to the LLM.
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
MAX_TOKENS = 6          # long messages are never treated as simple commands
SBERT_THRESHOLD = 0.65  # raised from 0.60: emotional sentences were hijacked by skills
DIFFLIB_THRESHOLD = 0.75


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
        if not t or len(t.split()) > MAX_TOKENS:
            return "none", 0.0
        if self._model is not None:
            q = self._model.encode([t], normalize_embeddings=True)[0]
            sims = self._emb @ q
            k = int(sims.argmax())
            score, intent = float(sims[k]), self.examples[k][0]
            return (intent, score) if score >= SBERT_THRESHOLD else ("none", score)
        best, best_i = 0.0, "none"
        toks = t.split()
        for intent, p in self.examples:
            s = difflib.SequenceMatcher(None, t, p).ratio()
            if len(p.split()) >= 2 and p in t:
                s = max(s, 0.9)
            if len(p.split()) == 1 and toks[:1] == [p] and len(toks) <= 2:
                s = max(s, 0.85)
            if s > best:
                best, best_i = s, intent
        return (best_i, best) if best >= DIFFLIB_THRESHOLD else ("none", best)