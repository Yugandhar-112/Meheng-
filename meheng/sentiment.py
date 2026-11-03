"""Sentiment + emotion analysis with swappable engines.

Engines
-------
vader        Lexicon/rule baseline (fast, no download, English only).
transformer  RoBERTa-family models from the Hugging Face Hub:
               * cardiffnlp/twitter-roberta-base-sentiment-latest  (English polarity)
               * cardiffnlp/twitter-xlm-roberta-base-sentiment     (multilingual incl. Hindi)
               * SamLowe/roberta-base-go_emotions                  (28 fine-grained emotions)
             For romanised Hinglish, the transformer output is blended with a small
             lexicon (hybrid approach).

`get_analyzer("auto")` tries the transformer engine and silently falls back to VADER
if `transformers`/`torch` are missing or the models cannot be downloaded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_HINGLISH_TOKENS = {
    "hai", "hain", "nahi", "nahin", "kya", "mujhe", "bahut", "acha", "accha", "kaise",
    "kaisa", "main", "mera", "meri", "tum", "aap", "ho", "tha", "thi", "kuch", "bhi",
    "aaj", "yaar", "bhai", "dukhi", "khush", "udaas", "thak", "gaya", "gayi", "raha", "rahi",
    "gussa", "badhiya", "pareshan", "mood", "aa", "mein", "hu", "hun", "tabiyet", "tabiyat",
    "kharaab", "kharab", "kitne", "baje", "batao", "sunao", "kaun", "kahan", "kab", "kaun", "samay", "din",
}

# Tiny romanised-Hindi sentiment lexicon (used to correct the multilingual model)
_HI_POS = {"accha", "acha", "achha", "khush", "badhiya", "mast", "maza", "zabardast", "shukriya", "pyaar"}
_HI_NEG = {"gussa", "dukhi", "udaas", "pareshan", "pareshaan", "thak", "thaka", "bura", "bekar",
           "tension", "darr", "akela", "rona", "kharaab", "kharab", "bimaar", "bimar", "dard"}
_HI_NEGATORS = {"nahi", "nahin", "na", "mat"}


def detect_lang(text: str) -> str:
    """Very small script/keyword heuristic: 'hi' (Hindi/Hinglish) or 'en'."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters) > 0.2:
        return "hi"
    tokens = re.findall(r"[a-z]+", text.lower())
    if sum(t in _HINGLISH_TOKENS for t in tokens) >= 2:
        return "hi"
    return "en"


def hinglish_lexicon(text: str) -> float:
    """Return +1 / -1 / 0 from a tiny romanised-Hindi lexicon (negation flips the sign)."""
    toks = re.findall(r"[a-z]+", text.lower())
    score = sum(t in _HI_POS for t in toks) - sum(t in _HI_NEG for t in toks)
    if any(t in _HI_NEGATORS for t in toks):
        score = -score
    return 1.0 if score > 0 else -1.0 if score < 0 else 0.0


@dataclass
class Sentiment:
    label: str                  # negative | neutral | positive
    confidence: float           # probability of `label` (0-1)
    compound: float             # signed polarity in [-1, 1]
    emotions: list[tuple[str, float]] = field(default_factory=list)  # top-k fine-grained
    engine: str = "vader"
    lang: str = "en"

    @property
    def top_emotion(self) -> str | None:
        return self.emotions[0][0] if self.emotions else None


class VaderEngine:
    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vs = SentimentIntensityAnalyzer()

    def analyze(self, text: str, lang: str = "en") -> Sentiment:
        s = self._vs.polarity_scores(text)
        c = s["compound"]
        label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        conf = {"positive": s["pos"], "negative": s["neg"], "neutral": s["neu"]}[label]
        return Sentiment(label, float(conf), float(c), [], self.name, lang)


class TransformerEngine:
    name = "transformer"
    EN_SENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ML_SENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    EMO = "SamLowe/roberta-base-go_emotions"
    _LABEL_FIX = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    _LEXICON_WEIGHT = 0.6   # weight of the Hinglish lexicon when it fires

    def __init__(self, use_emotions: bool = True):
        from transformers import pipeline  # raises ImportError if missing
        import torch
        self._pipeline = pipeline
        self._device = 0 if torch.cuda.is_available() else -1
        self._pipes: dict[str, object] = {}
        self._use_emotions = use_emotions
        self._warned_ml = False
        self._vader = None
        self._get("sent_en")                      # fail fast if the hub is unreachable
        if use_emotions:
            self._get("emo")
        try:                                      # preload so the first Hindi message is not slow
            self._get("sent_ml")
        except Exception as e:
            print(f"[meheng] multilingual model unavailable ({type(e).__name__}); "
                  f"Hindi/Hinglish will use VADER")
            self._warned_ml = True

    def _get(self, key: str):
        if key not in self._pipes:
            model = {"sent_en": self.EN_SENT, "sent_ml": self.ML_SENT, "emo": self.EMO}[key]
            self._pipes[key] = self._pipeline(
                "text-classification", model=model, top_k=None, device=self._device
            )
        return self._pipes[key]

    @staticmethod
    def _flat(out):
        return out[0] if out and isinstance(out[0], list) else out

    def analyze(self, text: str, lang: str = "en") -> Sentiment:
        key = "sent_ml" if lang != "en" else "sent_en"
        try:
            pipe = self._get(key)
        except Exception as e:
            if key == "sent_en":
                raise
            if not self._warned_ml:
                print(f"[meheng] multilingual model unavailable ({type(e).__name__}); "
                      f"using VADER for Hindi/Hinglish")
                self._warned_ml = True
            if self._vader is None:
                self._vader = VaderEngine()
            return self._vader.analyze(text, lang)

        probs = {}
        for d in self._flat(pipe(text, truncation=True, max_length=256)):
            lab = d["label"].lower()
            probs[self._LABEL_FIX.get(lab, lab)] = float(d["score"])

        # Hybrid correction for romanised Hinglish (low-resource for the transformer)
        if lang != "en":
            lex = hinglish_lexicon(text)
            if lex != 0.0:
                w = self._LEXICON_WEIGHT
                target = {"positive": float(lex > 0), "negative": float(lex < 0), "neutral": 0.0}
                probs = {k: (1 - w) * probs.get(k, 0.0) + w * target[k] for k in target}

        label = max(probs, key=probs.get)
        compound = probs.get("positive", 0.0) - probs.get("negative", 0.0)

        emotions: list[tuple[str, float]] = []
        if self._use_emotions and lang == "en":
            raw = self._flat(self._get("emo")(text, truncation=True, max_length=256))
            raw = sorted(raw, key=lambda d: d["score"], reverse=True)
            emotions = [(d["label"], float(d["score"])) for d in raw if d["label"] != "neutral"][:3]
        return Sentiment(label, probs[label], compound, emotions, self.name, lang)


def get_analyzer(prefer: str = "auto"):
    """Return an engine. prefer: 'auto' | 'transformer' | 'vader'."""
    if prefer in ("auto", "transformer"):
        try:
            return TransformerEngine()
        except Exception as e:  # ImportError, OSError (offline), etc.
            if prefer == "transformer":
                raise
            print(f"[meheng] transformer engine unavailable ({type(e).__name__}); using VADER fallback")
    return VaderEngine()