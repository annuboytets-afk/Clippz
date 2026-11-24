"""Tiny SQLite store for clips so approvals survive restarts."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    streamer TEXT NOT NULL,
    platform TEXT NOT NULL,
    title TEXT,
    path TEXT NOT NULL,
    preview_path TEXT,
    score REAL,
    caption TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | posting | posted | rejected | failed
    discord_message_id INTEGER,
    post_id TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS clips_msg ON clips(discord_message_id);
"""


class ClipDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add(self, **fields) -> int:
        fields.setdefault("created", time.time())
        cols = ",".join(fields)
        q = ",".join("?" * len(fields))
        return self.conn.execute(f"INSERT INTO clips ({cols}) VALUES ({q})", list(fields.values())).lastrowid

    def get(self, clip_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()

    def by_message(self, message_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM clips WHERE discord_message_id=?", (message_id,)).fetchone()

    def update(self, clip_id: int, **fields):
        sets = ",".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE clips SET {sets} WHERE id=?", [*fields.values(), clip_id])

    def transition(self, clip_id: int, from_status: str, to_status: str) -> bool:
        """Atomic status change; False if the clip was not in from_status (e.g. double click)."""
        cur = self.conn.execute("UPDATE clips SET status=? WHERE id=? AND status=?", (to_status, clip_id, from_status))
        return cur.rowcount == 1
