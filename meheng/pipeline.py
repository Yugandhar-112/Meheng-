"""Orchestrator: safety -> sentiment -> mood -> intent -> (skill | emotion-aware LLM) -> log."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime

from . import safety
from .intent import IntentMatcher
from .llm import FallbackBackend, build_system_prompt, get_backend
from .mood import MoodTracker
from .sentiment import Sentiment, detect_lang, get_analyzer
from .storage import Store

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    "Why did the neural network go to therapy? Too many unresolved layers.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
]

SKILL_TEXT = {
    "greeting": {"en": "Hi {name}! I'm Meheng. How are you feeling today?",
                 "hi": "Namaste {name}! Main Meheng hoon. Aap aaj kaisa mehsoos kar rahe hain?"},
    "goodbye": {"en": "Goodbye {name}! Take care of yourself.", "hi": "Alvida {name}! Apna khayal rakhiye."},
    "identity": {"en": "I'm Meheng, an emotion-aware AI assistant. I read the sentiment and emotion in what you write "
                       "and adapt my replies.",
                 "hi": "Main Meheng hoon, ek emotion-aware AI assistant. Main aapke shabdon ki bhavna samajhkar jawab deta hoon."},
    "help": {"en": "I can chat, notice how you're feeling, tell the time/date, tell jokes, and track your mood "
                   "over the conversation. Just talk to me!",
             "hi": "Main baat kar sakta hoon, aapka mood samajh sakta hoon, time/date bata sakta hoon aur jokes suna sakta hoon."},
}


@dataclass
class Turn:
    reply: str
    sentiment: Sentiment
    intent: str
    intent_score: float
    crisis: bool
    objectionable: bool
    backend: str
    latency_ms: float


class Meheng:
    def __init__(self, user: str = "guest", analyzer=None, matcher=None, llm=None, store=None,
                 history_window: int = 8):
        self.user = user
        self.analyzer = analyzer or get_analyzer("auto")
        self.matcher = matcher or IntentMatcher()
        self.llm = llm or get_backend("auto")
        self.store = store or Store()
        self.mood = MoodTracker()
        self.messages: list[dict] = []
        self.window = history_window

    def _skill(self, intent: str, lang: str) -> str | None:
        now = datetime.now()
        if intent == "time":
            return now.strftime("It's %I:%M %p.") if lang == "en" else now.strftime("Abhi %I:%M %p ho rahe hain.")
        if intent == "date":
            return now.strftime("Today is %A, %d %B %Y.") if lang == "en" else now.strftime("Aaj %A, %d %B %Y hai.")
        if intent == "joke":
            return random.choice(JOKES)
        if intent in SKILL_TEXT:
            return SKILL_TEXT[intent][lang].format(name=self.user)
        return None

    def respond(self, text: str) -> Turn:
        t0 = time.perf_counter()
        lang = detect_lang(text)
        sent = self.analyzer.analyze(text, lang)
        self.mood.update(sent.compound)
        crisis, rude = safety.is_crisis(text), safety.is_objectionable(text)
        intent, iscore = ("none", 0.0)
        backend = self.llm.name

        if crisis:                                    # 1) safety always wins
            reply, intent, backend = safety.CRISIS_REPLY[lang], "crisis", "safety"
        elif rude:                                    # 2) objectionable language
            reply, intent, backend = ("I'm happy to help, but let's keep our conversation respectful. "
                                      "What would you like to talk about?"), "objectionable", "safety"
        else:
            intent, iscore = self.matcher.match(text)
            skill = None
            # a greeting from someone clearly upset should get empathy, not a canned hello
            if not (intent == "greeting" and sent.compound < -0.4) and intent != "none":
                skill = self._skill(intent, lang)
            if skill:
                reply, backend = skill, "skill"
            else:
                intent = "none" if intent in ("greeting",) else intent
                msgs = self.messages[-self.window:] + [{"role": "user", "content": text}]
                system = build_system_prompt(sent, self.mood.summary())
                try:
                    if isinstance(self.llm, FallbackBackend):
                        reply = self.llm.generate(system, msgs, sentiment=sent)
                    else:
                        reply = self.llm.generate(system, msgs)
                except Exception as e:                # network/API errors must not crash the chat
                    print(f"[meheng] LLM error: {e}")
                    reply, backend = FallbackBackend().generate(system, msgs, sentiment=sent), "fallback"

        self.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.store.log(self.user, text, reply, sent, intent, rude, crisis)
        return Turn(reply, sent, intent, iscore, crisis, rude, backend, (time.perf_counter() - t0) * 1000)
