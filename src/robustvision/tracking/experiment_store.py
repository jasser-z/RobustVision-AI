from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, status TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def add(self, payload: dict, status="completed") -> int:
        with sqlite3.connect(self.path) as c:
            cur = c.execute(
                "INSERT INTO experiments(status,payload) VALUES (?,?)",
                (status, json.dumps(payload)),
            )
            return cur.lastrowid

    def list(self) -> list[dict]:
        with sqlite3.connect(self.path) as c:
            rows = c.execute(
                "SELECT id,created_at,status,payload FROM experiments ORDER BY id DESC"
            ).fetchall()
        return [{"id": x[0], "created_at": x[1], "status": x[2], **json.loads(x[3])} for x in rows]
