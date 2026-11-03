"""Running mood tracker: exponential moving average + short-term trend."""
from __future__ import annotations


class MoodTracker:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.ema: float | None = None
        self.history: list[float] = []

    def update(self, compound: float) -> None:
        self.history.append(compound)
        self.ema = compound if self.ema is None else self.alpha * compound + (1 - self.alpha) * self.ema

    @property
    def trend(self) -> str:
        h = self.history
        if len(h) < 4:
            return "unknown"
        recent, before = h[-3:], h[-6:-3]
        d = sum(recent) / len(recent) - sum(before) / len(before)
        return "improving" if d > 0.25 else "declining" if d < -0.25 else "steady"

    @property
    def level(self) -> str:
        if self.ema is None:
            return "unknown"
        return "positive" if self.ema > 0.3 else "low" if self.ema < -0.3 else "mixed"

    def summary(self) -> str:
        if self.ema is None:
            return "No mood information yet."
        return f"Overall mood so far: {self.level} (score {self.ema:+.2f}); recent trend: {self.trend}."
