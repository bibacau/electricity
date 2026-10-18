import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

DEFAULT_SETTINGS = {
    "name": "Atlas", "style": "Clear, thoughtful and candid", "language": "English",
    "detail": "balanced", "model": "", "embedding_model": "", "remember_chat": True,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, role TEXT NOT NULL, content TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0, embedding TEXT, embedding_model TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE, rating INTEGER NOT NULL, correction TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, detail TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, status TEXT NOT NULL, config TEXT NOT NULL, result TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS checkpoints (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, path TEXT NOT NULL, baseline REAL NOT NULL, candidate REAL NOT NULL, evidence TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0, created TEXT DEFAULT CURRENT_TIMESTAMP);
"""


class Store:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "workbench.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript(SCHEMA)
            db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (json.dumps(DEFAULT_SETTINGS),))

    @contextmanager
    def connect(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args)]

    def execute(self, sql, args=()):
        with self.connect() as db:
            cur = db.execute(sql, args)
            return cur.lastrowid

    def settings(self):
        return DEFAULT_SETTINGS | json.loads(self.rows("SELECT value FROM settings WHERE id=1")[0]["value"])

    def event(self, kind, detail):
        self.execute("INSERT INTO events(kind,detail) VALUES (?,?)", (kind, json.dumps(detail, ensure_ascii=False)))

    def backup(self):
        target = self.directory / "backups"
        target.mkdir(exist_ok=True)
        from uuid import uuid4
        path = target / f"memory-{uuid4().hex}.sqlite3"
        with self.connect() as source:
            with sqlite3.connect(path) as destination:
                source.backup(destination)
        return path
