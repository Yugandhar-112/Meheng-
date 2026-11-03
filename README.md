# Meheng-NLP — an Emotion-Aware Conversational Assistant

Natural Language Processing project. Evolution of the *Meheng.AI* voice assistant (Class XII project): the
rule/command assistant is rebuilt as a **text chatbot that understands how the user feels** and adapts its replies.

## What it does
| Stage | Technique | File |
|---|---|---|
| 1. Safety | High-recall crisis detection (EN/Hindi/Hinglish) → fixed helpline reply *before* any model runs; profanity flagging (replaces "objectionable query tracking" from the original) | `meheng/safety.py` |
| 2. Sentiment + emotion | **RoBERTa** (`twitter-roberta-base-sentiment-latest`) for polarity, **GoEmotions RoBERTa** for 28 fine-grained emotions, **XLM-RoBERTa** (`twitter-xlm-roberta-base-sentiment`) for Hindi/Hinglish. VADER = baseline / offline fallback | `meheng/sentiment.py` |
| 3. Mood tracking | Exponential moving average of polarity + trend (improving/steady/declining) across the conversation | `meheng/mood.py` |
| 4. Intent | **Sentence-BERT** (multilingual MiniLM) embeddings + cosine similarity to example phrases → built-in skills (time, date, jokes…) with a `difflib` fallback | `meheng/intent.py` |
| 5. Generation | **LLM with emotion-conditioned prompting**: detected emotion, polarity and mood trend are injected into the system prompt (Claude or Gemini). Offline template responder if no API key | `meheng/llm.py` |
| 6. Logging | SQLite `conversation_logs` (query, response, sentiment, emotion, intent, flags) — same idea as the original MySQL ConversationLogs table | `meheng/storage.py` |
| 7. Evaluation | Accuracy / macro-F1 / confusion matrix, per language, VADER vs transformer | `eval/eval_sentiment.py` |

```
user text → language detect → SAFETY ──crisis/rude──► fixed safe reply
                    │                                       │
                    ▼                                       │
        sentiment + emotion (RoBERTa / XLM-R) → mood tracker│
                    ▼                                       │
          intent match (SBERT) ─skill─► time/date/joke…     │
                    │ none                                  │
                    ▼                                       ▼
     emotion-conditioned LLM prompt ──► reply ──► SQLite log ──► UI (badges + mood chart)
```

## Run it
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# optional, for a real LLM (otherwise offline fallback is used):
export ANTHROPIC_API_KEY=...     # or GEMINI_API_KEY=...   (Windows: set VAR=...)
streamlit run app.py             # web UI
python cli.py                    # terminal version
python -m pytest -q tests        # unit tests
python eval/eval_sentiment.py --engines vader transformer   # produces eval/results.md
```
First run downloads the Hugging Face models (~1.5 GB; +~1.1 GB the first time a Hindi message appears).
If a download fails the app automatically falls back to VADER and tells you in the sidebar.

## Results (custom 45-sentence set: 32 English, 13 Hindi/Hinglish)
| Engine | Accuracy | Macro-F1 | English | Hindi/Hinglish |
|---|---|---|---|---|
| VADER (baseline) | 0.667 | 0.667 | 0.812 | 0.308 |
| RoBERTa / XLM-R transformer | *run `eval_sentiment.py` and paste here* | | | |

VADER collapses on Hindi/Hinglish (no vocabulary) — the motivation for the multilingual transformer.
**The set is small and hand-labelled, so treat it as a demonstration, not a benchmark.** For a more credible
number run `--tweet-eval 500` (TweetEval sentiment test split).

## Limitations (be upfront about these)
- Sentiment ≠ emotion ≠ mental state. This is **not a diagnostic or therapeutic tool**.
- Crisis detection is keyword-based (high recall, many false positives, misses indirect phrasing).
- Profanity list is English-only. Sarcasm ("Oh great, another Monday") remains hard for all engines.
- Romanised Hinglish is handled by XLM-R only approximately; fine-grained emotions are English-only.
- LLM replies can be wrong; no retrieval/grounding is used.

## Future work
Fine-tune **ModernBERT** (Warner et al., 2024) on GoEmotions + Hindi data; add retrieval-augmented generation (RAG);
speech input/output (Whisper + TTS) to restore the original voice-assistant feature; user login/admin panel.

## References
- Vaswani et al., 2017 — *Attention Is All You Need*
- Liu et al., 2019 — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Conneau et al., 2020 — *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R)
- Barbieri et al., 2022 — *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*
- Loureiro et al., 2022 — *TimeLMs: Diachronic Language Models from Twitter*
- Demszky et al., 2020 — *GoEmotions: A Dataset of Fine-Grained Emotions*
- Reimers & Gurevych, 2019 — *Sentence-BERT*
- Li et al., 2023 — *Large Language Models Understand and Can Be Enhanced by Emotional Stimuli* (EmotionPrompt)
- Hutto & Gilbert, 2014 — *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*
- Warner et al., 2024 — *Smarter, Better, Faster, Longer: ModernBERT*
