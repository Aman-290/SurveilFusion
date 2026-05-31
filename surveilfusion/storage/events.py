import json
import sqlite3
from pathlib import Path

from surveilfusion.core.models import SurveillanceEvent


class EventStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL
                )
                """
            )

    def add(self, event: SurveillanceEvent) -> None:
        payload = event.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO events
                (id, camera_id, kind, severity, title, summary, created_at, acknowledged, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.camera_id,
                    event.kind.value,
                    event.severity.value,
                    event.title,
                    event.summary,
                    event.created_at.isoformat(),
                    int(event.acknowledged),
                    payload,
                ),
            )

    def latest(self, limit: int = 50) -> list[SurveillanceEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [SurveillanceEvent.model_validate(json.loads(row["payload"])) for row in rows]

    def acknowledge(self, event_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                return False
            event = SurveillanceEvent.model_validate(json.loads(row["payload"]))
            event.acknowledged = True
            self.add(event)
        return True
