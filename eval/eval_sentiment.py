"""Compare sentiment engines on a labelled set.

    python eval/eval_sentiment.py                       # built-in 45-sentence set (en + hi/hinglish)
    python eval/eval_sentiment.py --engines vader transformer
    python eval/eval_sentiment.py --tweet-eval 500      # also run on 500 TweetEval-sentiment test tweets (needs `datasets`)

Writes eval/results.md with accuracy, macro-F1, per-language accuracy and confusion matrices.
"""
import argparse
import pathlib
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from meheng.sentiment import VaderEngine, TransformerEngine, detect_lang  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


def predict(engine, df, use_detected_lang=True):
    return [engine.analyze(t, detect_lang(t) if use_detected_lang else l).label
            for t, l in zip(df.text, df.lang)]


def report(name, df, preds, out):
    acc = accuracy_score(df.label, preds)
    f1 = f1_score(df.label, preds, labels=LABELS, average="macro")
    out.append(f"### {name}\n\n- Accuracy: **{acc:.3f}**  |  Macro-F1: **{f1:.3f}**  |  n={len(df)}")
    for lang, sub in df.assign(pred=preds).groupby("lang"):
        out.append(f"- Accuracy on `{lang}`: {accuracy_score(sub.label, sub.pred):.3f} (n={len(sub)})")
    cm = pd.DataFrame(confusion_matrix(df.label, preds, labels=LABELS),
                      index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    out.append("\n```\n" + cm.to_string() + "\n```\n")
    print(f"{name:12s} acc={acc:.3f} macroF1={f1:.3f} (n={len(df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["vader", "transformer"])
    ap.add_argument("--tweet-eval", type=int, default=0)
    a = ap.parse_args()

    engines = {}
    for n in a.engines:
        try:
            engines[n] = VaderEngine() if n == "vader" else TransformerEngine(use_emotions=False)
        except Exception as e:
            print(f"skipping {n}: {type(e).__name__}: {e}")

    out = ["# Sentiment evaluation results\n"]
    df = pd.read_csv(ROOT / "test_set.csv")
    out.append("## Custom set (English + Hindi/Hinglish)\n")
    for n, e in engines.items():
        report(n, df, predict(e, df), out)

    if a.tweet_eval:
        from datasets import load_dataset
        ds = load_dataset("tweet_eval", "sentiment", split="test").shuffle(seed=0).select(range(a.tweet_eval))
        tw = pd.DataFrame({"text": ds["text"], "label": [LABELS[i] for i in ds["label"]], "lang": "en"})
        out.append(f"## TweetEval sentiment (test sample of {a.tweet_eval})\n")
        for n, e in engines.items():
            report(n, tw, predict(e, tw, use_detected_lang=False), out)

    (ROOT / "results.md").write_text("\n".join(out))
    print(f"\nSaved {ROOT/'results.md'}")


if __name__ == "__main__":
    main()
