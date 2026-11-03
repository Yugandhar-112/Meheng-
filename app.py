"""Streamlit UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from meheng import Meheng
from meheng.intent import IntentMatcher
from meheng.llm import get_backend
from meheng.sentiment import get_analyzer
from meheng.storage import Store

st.set_page_config(page_title="Meheng-NLP", page_icon="🧠", layout="wide")

EMOJI = {"joy": "😄", "amusement": "😂", "excitement": "🤩", "love": "🥰", "gratitude": "🙏", "optimism": "🌈",
         "admiration": "👏", "approval": "👍", "pride": "😌", "relief": "😮‍💨", "caring": "🤗", "desire": "✨",
         "sadness": "😢", "grief": "💔", "disappointment": "😞", "remorse": "😔", "embarrassment": "😳",
         "fear": "😨", "nervousness": "😰", "anger": "😠", "annoyance": "😒", "disgust": "🤢",
         "disapproval": "👎", "confusion": "😕", "curiosity": "🤔", "surprise": "😲", "realization": "💡"}
POL = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}


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
    emo = f" · {EMOJI.get(s.top_emotion, '💬')} {s.top_emotion} ({s.emotions[0][1]:.0%})" if s.emotions else ""
    meta = (f"{POL[s.label]} {s.label} ({s.compound:+.2f}){emo} · intent: {t.intent} · "
            f"{s.engine}/{t.backend} · {t.latency_ms:.0f} ms" + (" · ⚠️ flagged" if t.objectionable else ""))
    st.session_state.chat += [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": t.reply, "meta": meta}]
    st.rerun()
