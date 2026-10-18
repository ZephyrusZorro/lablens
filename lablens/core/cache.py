"""Answer cache - the same "which pin is ground?" asked 50 times in a lab should cost one LLM call.

Keyed by (component|part, normalised bag-of-words question). SQLite so it survives restarts and can
be shared by every bench on a lab network drive.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from lablens.types import Answer, Citation


class AnswerCache:
    def __init__(self, path: Path, ttl_days: float = 30):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self.ttl = ttl_days * 86400
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (ctx TEXT, q TEXT, answer TEXT, cites TEXT,"
                         " backend TEXT, ts REAL, hits INTEGER DEFAULT 0, PRIMARY KEY (ctx, q))")
        self._db.commit()

    def get(self, key: Tuple[str, str]) -> Optional[Answer]:
        with self._lock:
            row = self._db.execute("SELECT answer, cites, backend, ts FROM answers WHERE ctx=? AND q=?",
                                   key).fetchone()
            if not row or time.time() - row[3] > self.ttl:
                return None
            self._db.execute("UPDATE answers SET hits = hits + 1 WHERE ctx=? AND q=?", key)
            self._db.commit()
        cites = [Citation(**c) for c in json.loads(row[1])]
        return Answer(row[0], cites, row[2], 0.0, cached=True)

    def put(self, key: Tuple[str, str], ans: Answer) -> None:
        if not ans.citations:  # only cache grounded answers
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?,0)",
                             (*key, ans.text, json.dumps([c.__dict__ for c in ans.citations]), ans.backend,
                              time.time()))
            self._db.commit()

    def top(self, n: int = 10):
        with self._lock:
            return self._db.execute("SELECT ctx, q, hits FROM answers ORDER BY hits DESC LIMIT ?", (n,)).fetchall()
