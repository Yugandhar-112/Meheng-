import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from meheng import Meheng
from meheng.intent import IntentMatcher
from meheng.mood import MoodTracker
from meheng.safety import is_crisis, is_objectionable
from meheng.sentiment import VaderEngine, detect_lang
from meheng.storage import Store


@pytest.fixture
def bot():
    return Meheng(user="t", analyzer=VaderEngine(), matcher=IntentMatcher(use_embeddings=False), store=Store(":memory:"))


def test_lang_detection():
    assert detect_lang("आज मैं बहुत खुश हूँ") == "hi"
    assert detect_lang("aaj main bahut dukhi hoon yaar") == "hi"
    assert detect_lang("I am happy today") == "en"


def test_sentiment_polarity():
    v = VaderEngine()
    assert v.analyze("I love this, it's wonderful!").label == "positive"
    assert v.analyze("This is awful and I hate it").label == "negative"


def test_crisis_and_objectionable():
    assert is_crisis("I want to die") and is_crisis("mujhe khudkushi ke khayal aate hain")
    assert not is_crisis("what a lovely day")
    assert is_objectionable("you fucking idiot") and not is_objectionable("hello there")


def test_intents(bot):
    m = bot.matcher
    assert m.match("what time is it")[0] == "time"
    assert m.match("tell me a joke")[0] == "joke"
    assert m.match("I had a really long and difficult day at college today")[0] == "none"


def test_mood_trend():
    m = MoodTracker()
    for c in [0.8, 0.7, 0.6, -0.5, -0.6, -0.7]:
        m.update(c)
    assert m.trend == "declining" and m.level in {"mixed", "low"}


def test_pipeline_flow_and_logging(bot):
    assert bot.respond("hello").backend == "skill"
    assert bot.respond("I want to die").crisis
    assert bot.respond("you fucking idiot").objectionable
    t = bot.respond("I feel so sad and lonely")
    assert t.sentiment.label == "negative" and t.backend == "fallback"
    assert len(bot.store.history("t")) == 4
    assert bot.store.export_csv("t").startswith("log_id")
