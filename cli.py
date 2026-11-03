"""Terminal chat:  python cli.py [--engine auto|vader|transformer] [--name Ananya]"""
import argparse

from meheng import Meheng
from meheng.sentiment import get_analyzer

p = argparse.ArgumentParser()
p.add_argument("--engine", default="auto", choices=["auto", "vader", "transformer"])
p.add_argument("--name", default="friend")
a = p.parse_args()

bot = Meheng(user=a.name, analyzer=get_analyzer(a.engine))
print(f"Meheng ready  [sentiment={bot.analyzer.name} | intents={bot.matcher.mode} | llm={bot.llm.name}]  (type 'quit' to exit)\n")
while True:
    try:
        text = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if text.lower() in {"quit", "exit"}:
        break
    if not text:
        continue
    t = bot.respond(text)
    emo = f" {t.sentiment.top_emotion}" if t.sentiment.top_emotion else ""
    print(f"meheng > {t.reply}")
    print(f"         [{t.sentiment.label} {t.sentiment.compound:+.2f}{emo} | intent={t.intent} | {t.latency_ms:.0f} ms]\n")
print(bot.mood.summary())
