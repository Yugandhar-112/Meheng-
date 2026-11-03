"""SQLite conversation log (mirrors the ConversationLogs table in the original Meheng design)."""
from __future__ import annotations

import csv
import io
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    query           TEXT NOT NULL,
    response        TEXT,
    sentiment       TEXT,
    compound        REAL,
    top_emotion     TEXT,
    intent          TEXT,
    engine          TEXT,
    is_objectionable INTEGER DEFAULT 0,
    is_crisis        INTEGER DEFAULT 0,
    log_time        TEXT DEFAULT CURRENT_TIMESTAMP
)"""


class Store:
    def __init__(self, path: str = "meheng.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def log(self, user_id: str, query: str, response: str, s, intent: str,
            objectionable: bool, crisis: bool) -> None:
        self.conn.execute(
            "INSERT INTO conversation_logs (user_id, query, response, sentiment, compound, top_emotion, "
            "intent, engine, is_objectionable, is_crisis) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user_id, query, response, s.label, s.compound, s.top_emotion, intent, s.engine,
             int(objectionable), int(crisis)))
        self.conn.commit()

    def history(self, user_id: str, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM conversation_logs WHERE user_id=? ORDER BY log_id DESC LIMIT ?",
            (user_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def export_csv(self, user_id: str) -> str:
        rows = self.history(user_id, 10_000)
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        return buf.getvalue()
