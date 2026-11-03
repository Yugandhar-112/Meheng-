"""Streamlit UI:  streamlit run app.py"""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from meheng import Meheng
from meheng.intent import IntentMatcher
from meheng.llm import get_backend
from meheng.sentiment import get_analyzer
from meheng.storage import Store

st.set_page_config(page_title="Meheng-NLP", page_icon="🧠", layout="wide")

# Logs are saved next to app.py, i.e. inside the project folder (D:\meheng_nlp\logs)
LOG_DIR = Path(__file__).resolve().parent / "logs"

EMOJI = {"joy": "😄", "amusement": "😂", "excitement": "🤩", "love": "🥰", "gratitude": "🙏", "optimism": "🌈",
         "admiration": "👏", "approval": "👍", "pride": "😌", "relief": "😮‍💨", "caring": "🤗", "desire": "✨",
         "sadness": "😢", "grief": "💔", "disappointment": "😞", "remorse": "😔", "embarrassment": "😳",
         "fear": "😨", "nervousness": "😰", "anger": "😠", "annoyance": "😒", "disgust": "🤢",
         "disapproval": "👎", "confusion": "😕", "curiosity": "🤔", "surprise": "😲", "realization": "💡"}
POL = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}


def save_logs_to_project(csv_text: str, user: str) -> Path:
    """Write the CSV into <project>/logs/ and return the file path."""
    LOG_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", user).strip("_") or "user"
    path = LOG_DIR / f"meheng_logs_{safe}.csv"
    # utf-8-sig so Excel shows Hindi text correctly; newline="" avoids blank rows on Windows
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(csv_text)
    return path


@st.cache_resource(show_spinner="Loading NLP models (first run downloads them)...")
def load_components(engine: str):
    return get_analyzer(engine), IntentMatcher(), get_backend("auto"), Store("meheng.db")


with st.sidebar:
    st.title("🧠 Meheng-NLP")
    name = st.text_input("Your name", "friend")
    engine = st.selectbox("Sentiment engine", ["auto", "transformer", "vader"],
                          help="auto = RoBERTa transformers if available, else VADER baseline")

analyzer, matcher, llm, store = load_components(engine)

if st.session_state.get("_key") != (name, engine):
    st.session_state._key = (name, engine)
    st.session_state.bot = Meheng(user=name, analyzer=analyzer, matcher=matcher, llm=llm, store=store)
    st.session_state.chat = []
bot: Meheng = st.session_state.bot

with st.sidebar:
    st.caption(f"Sentiment: **{analyzer.name}** · Intents: **{matcher.mode}** · LLM: **{llm.name}**")
    st.subheader("Mood timeline")
    if bot.mood.history:
        st.line_chart(pd.DataFrame({"polarity (-1..1)": bot.mood.history}))
        st.info(bot.mood.summary())
    else:
        st.write("Start chatting to see your mood curve.")
    csv = store.export_csv(name)
    if csv:
        if st.button("Save logs to project folder"):
            try:
                saved = save_logs_to_project(csv, name)
                st.success(f"Saved to {saved}")
            except OSError as e:
                st.error(f"Could not save logs: {e}")
        st.download_button("Download my logs (CSV)", csv, "meheng_logs.csv", "text/csv")
    st.caption("Not a medical tool. If you're in distress in India, call Tele-MANAS 14416.")

st.header("Chat with Meheng")
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

if prompt := st.chat_input("Say something..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("thinking..."):
        t = bot.respond(prompt)
    s = t.sentiment
    emo = (f" · {EMOJI.get(s.top_emotion, '💬')} {s.top_emotion} ({s.emotions[0][1]:.0%})"
           if s.emotions and s.emotions[0][1] >= 0.20 else "")
    meta = (f"{POL[s.label]} {s.label} ({s.compound:+.2f}){emo} · intent: {t.intent} · "
            f"{s.engine}/{t.backend} · {t.latency_ms:.0f} ms" + (" · ⚠️ flagged" if t.objectionable else ""))
    st.session_state.chat += [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": t.reply, "meta": meta}]
    st.rerun()