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

st.set_page_config(page_title="Meheng · Emotion-aware chatbot",
                   page_icon=":material/psychology:", layout="centered")

# Logs are saved next to app.py, i.e. inside the project folder (D:\meheng_nlp\logs)
LOG_DIR = Path(__file__).resolve().parent / "logs"

EMOJI = {"joy": "😄", "amusement": "😂", "excitement": "🤩", "love": "🥰", "gratitude": "🙏", "optimism": "🌈",
         "admiration": "👏", "approval": "👍", "pride": "😌", "relief": "😮‍💨", "caring": "🤗", "desire": "✨",
         "sadness": "😢", "grief": "💔", "disappointment": "😞", "remorse": "😔", "embarrassment": "😳",
         "fear": "😨", "nervousness": "😰", "anger": "😠", "annoyance": "😒", "disgust": "🤢",
         "disapproval": "👎", "confusion": "😕", "curiosity": "🤔", "surprise": "😲", "realization": "💡"}
BADGE = {"positive": "green", "neutral": "gray", "negative": "red"}
EXAMPLES = ["I am so happy today!", "mujhe bahut gussa aa raha hai", "what time is it", "tell me a joke"]

# Only gap the theme can't cover: a branded header banner (flat colour, works in light and dark mode)
CSS = """
<style>
.hero {display:flex; align-items:center; gap:16px; padding:20px 24px; margin-bottom:12px;
       border-radius:16px; background:#0F766E; color:#fff;}
.hero-icon {font-size:2.4rem; line-height:1;}
.hero h1 {margin:0; padding:0; font-size:1.8rem; color:#fff;}
.hero p {margin:2px 0 0; font-size:.95rem; color:#E6FFFB;}
@media (max-width: 640px) {
  .hero {padding:14px 16px; gap:12px;}
  .hero h1 {font-size:1.4rem;}
  .hero p {font-size:.82rem;}
}
</style>
"""


def safe(text: str) -> str:
    """Display-only: stop $...$ from rendering as LaTeX (stored/exported data is untouched)."""
    return text.replace("$", "＄")


def button_row():
    """Horizontal button group (falls back to a plain container on older Streamlit)."""
    try:
        return st.container(horizontal=True)
    except TypeError:
        return st.container()


def save_logs_to_project(csv_text: str, user: str) -> Path:
    """Write the CSV into <project>/logs/ and return the file path."""
    LOG_DIR.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", user).strip("_") or "user"
    path = LOG_DIR / f"meheng_logs_{name}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: Hindi opens fine in Excel
        f.write(csv_text)
    return path


def mood_chart(history: list[float]) -> None:
    """Line chart with the y-axis fixed to -1..1 (falls back to the plain chart)."""
    df = pd.DataFrame({"message": range(1, len(history) + 1), "polarity": history})
    try:
        import altair as alt
        chart = (alt.Chart(df)
                 .mark_line(point=True, color="#14B8A6", strokeWidth=2.5)
                 .encode(x=alt.X("message:Q", axis=alt.Axis(tickMinStep=1)),
                         y=alt.Y("polarity:Q", scale=alt.Scale(domain=[-1, 1])))
                 .properties(height=180))
        try:
            st.altair_chart(chart, width="stretch")
        except TypeError:
            st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.line_chart(df.set_index("message"))


@st.cache_resource(show_spinner="Loading NLP models (first run downloads them)...")
def load_components(engine: str):
    return get_analyzer(engine), IntentMatcher(), get_backend("auto"), Store("meheng.db")


def ensure_bot(name: str, engine: str, components) -> None:
    """(Re)create the bot when the name/engine changes or the chat was cleared."""
    if st.session_state.get("_key") != (name, engine):
        analyzer, matcher, llm, store = components
        st.session_state._key = (name, engine)
        st.session_state.bot = Meheng(user=name, analyzer=analyzer, matcher=matcher, llm=llm, store=store)
        st.session_state.chat = []


def reset_chat() -> None:
    st.session_state._key = None          # next run builds a fresh bot, empty chat and new mood curve


def format_meta(t) -> str:
    s = t.sentiment
    parts = [f":{BADGE[s.label]}-badge[{s.label} {s.compound:+.2f}]"]
    if s.emotions and s.emotions[0][1] >= 0.20:
        parts.append(f"{EMOJI.get(s.top_emotion, '')} {s.top_emotion} {s.emotions[0][1]:.0%}")
    parts += [f"intent: {t.intent}", f"{s.engine}/{t.backend}", f"{t.latency_ms:.0f} ms"]
    meta = " · ".join(parts)
    if t.objectionable:
        meta += " :orange-badge[:material/flag: flagged]"
    return meta


def send(prompt: str) -> None:
    t = st.session_state.bot.respond(prompt)
    st.session_state.chat += [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": t.reply, "meta": format_meta(t)}]


def render_sidebar_settings() -> tuple[str, str]:
    with st.sidebar:
        st.subheader(":material/tune: Settings")
        name = st.text_input("Your name", "friend")
        engine = st.selectbox("Sentiment engine", ["auto", "transformer", "vader"],
                              help="auto = RoBERTa transformers if available, else VADER baseline")
    return name, engine


def render_sidebar_footer(analyzer, matcher, llm) -> None:
    with st.sidebar:
        st.caption(f"Sentiment: **{analyzer.name}** · Intents: **{matcher.mode}** · LLM: **{llm.name}**")
        st.button("Clear chat", icon=":material/delete_sweep:", on_click=reset_chat)
        st.caption("Not a medical tool. If you're in distress in India, call Tele-MANAS 14416.")


def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="hero-icon">🧠</div><div><h1>Meheng</h1>'
        '<p>Your emotion-aware companion · English · हिन्दी · Hinglish</p></div></div>',
        unsafe_allow_html=True)


def render_mood(bot: Meheng) -> None:
    hist = bot.mood.history
    with st.expander(":material/monitoring: Mood timeline", expanded=bool(hist)):
        if not hist:
            st.write("Start chatting to see your mood curve.")
            return
        left, right = st.columns([1, 3], vertical_alignment="center")
        delta = f"{hist[-1] - hist[-2]:+.2f}" if len(hist) > 1 else None
        left.metric("Latest polarity", f"{hist[-1]:+.2f}", delta=delta)
        with right:
            mood_chart(hist)
        st.info(bot.mood.summary())


def render_data(store: Store, name: str) -> None:
    with st.expander(":material/folder_open: Your data"):
        csv = store.export_csv(name)
        if not csv:
            st.write("No messages logged yet.")
            return
        with button_row():
            save = st.button("Save to project folder", icon=":material/save:", type="primary")
            st.download_button("Download CSV", csv, "meheng_logs.csv", "text/csv",
                               icon=":material/download:")
        if save:
            try:
                st.success(f"Saved to {save_logs_to_project(csv, name)}")
            except OSError as e:
                st.error(f"Could not save logs: {e}")


def render_welcome() -> None:
    with st.container(border=True):
        st.subheader("Welcome")
        st.write("Tell me how your day is going. I read the emotion in your words and adapt my replies. "
                 "Try an example:")
        with button_row():
            for i, ex in enumerate(EXAMPLES):
                st.button(ex, key=f"example_{i}", on_click=send, args=(ex,))


def render_history() -> None:
    for m in st.session_state.chat:
        avatar = ":material/person:" if m["role"] == "user" else ":material/psychology:"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(safe(m["content"]))
            if m.get("meta"):
                st.caption(m["meta"])


def main() -> None:
    name, engine = render_sidebar_settings()
    components = load_components(engine)
    ensure_bot(name, engine, components)
    analyzer, matcher, llm, store = components
    render_sidebar_footer(analyzer, matcher, llm)

    prompt = st.chat_input("Say something...")      # always pinned to the bottom of the page
    if prompt:
        with st.spinner("Meheng is thinking..."):
            send(prompt)

    render_header()
    render_mood(st.session_state.bot)
    render_data(store, name)
    if not st.session_state.chat:
        render_welcome()
    render_history()


main()