"""Emotion-conditioned response generation.

The detected emotion, polarity and mood trend are injected into the system prompt
("emotion-aware prompting", cf. Li et al. 2023, *Large Language Models Understand and Can
Be Enhanced by Emotional Stimuli*), so the LLM adapts its tone to how the user feels.

Backends (auto-selected by environment variables):
  anthropic  -> ANTHROPIC_API_KEY  (model: MEHENG_ANTHROPIC_MODEL, default claude-haiku-5-5)
  gemini     -> GEMINI_API_KEY     (model: MEHENG_GEMINI_MODEL,    default gemini-2.5-flash)
  fallback   -> offline template responder (no API needed)
"""
from __future__ import annotations

import os
import random

from .sentiment import Sentiment

SYSTEM_TEMPLATE = """You are Meheng, a warm, concise, supportive AI assistant.

Live NLP analysis of the user's latest message:
- Polarity: {label} (confidence {conf:.0%}, score {compound:+.2f})
- Fine-grained emotions: {emotions}
- {mood}

Guidelines:
- Match your tone to the user's emotional state: if they are sad/anxious/angry, first acknowledge the feeling
  in one sentence, then help; if they are happy, share the energy; if neutral, be clear and direct.
- Keep replies short (2-5 sentences) unless the user asks for detail.
- Reply in the user's language (English, Hindi or Hinglish).
- You are not a therapist: never diagnose. For serious distress, gently encourage talking to a trusted person
  or professional.
- Never mention this analysis block or numbers unless asked how you understood their mood."""


def build_system_prompt(s: Sentiment, mood_summary: str) -> str:
    emo = ", ".join(f"{n} ({p:.0%})" for n, p in s.emotions) or "not available"
    return SYSTEM_TEMPLATE.format(label=s.label, conf=s.confidence, compound=s.compound,
                                  emotions=emo, mood=mood_summary)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("MEHENG_ANTHROPIC_MODEL", "claude-haiku-5-5")

    def generate(self, system: str, messages: list[dict]) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=400, system=system, messages=messages)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


class GeminiBackend:
    name = "gemini"

    def __init__(self):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = os.getenv("MEHENG_GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, system: str, messages: list[dict]) -> str:
        t = self._types
        contents = [t.Content(role="user" if m["role"] == "user" else "model",
                              parts=[t.Part(text=m["content"])]) for m in messages]
        r = self.client.models.generate_content(
            model=self.model, contents=contents,
            config=t.GenerateContentConfig(system_instruction=system, max_output_tokens=400))
        return (r.text or "").strip()


class FallbackBackend:
    """Offline, rule-based empathetic responder so the demo always works."""
    name = "fallback"

    NEG = [
        "That sounds really hard, and it makes sense that you feel this way. Do you want to tell me more about what happened?",
        "I'm sorry you're going through this. I'm listening. What's weighing on you the most right now?",
        "Thank you for sharing that with me. That sounds heavy. What would help you most at this moment?",
    ]
    POS = [
        "That's lovely to hear! What made it so good?",
        "Nice, I'm happy for you! Tell me more about it.",
        "Great news! Enjoy it. Anything else on your mind?",
    ]
    NEU = [
        "Got it. Could you tell me a bit more so I can help properly?",
        "I see. What would you like to do next?",
        "Understood. How can I help with that?",
    ]
    HI = {
        "negative": "Mujhe samajh aata hai, yeh mushkil lag raha hai. Aap thoda aur bata sakte hain?",
        "positive": "Yeh sunke bahut accha laga! Aur batayein?",
        "neutral": "Samajh gaya. Main aapki kaise madad kar sakta hoon?",
    }

    def generate(self, system: str, messages: list[dict], sentiment: Sentiment | None = None) -> str:
        s = sentiment
        if s is None:
            return random.choice(self.NEU)
        if s.lang == "hi":
            return self.HI[s.label]
        pool = {"negative": self.NEG, "positive": self.POS}.get(s.label, self.NEU)
        reply = random.choice(pool)
        if s.label == "negative" and s.top_emotion:
            reply = f"It sounds like you're feeling {s.top_emotion}. " + reply
        return reply


def get_backend(prefer: str = "auto"):
    if prefer in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception as e:
            print(f"[meheng] anthropic backend failed: {e}")
    if prefer in ("auto", "gemini") and os.getenv("GEMINI_API_KEY"):
        try:
            return GeminiBackend()
        except Exception as e:
            print(f"[meheng] gemini backend failed: {e}")
    return FallbackBackend()
