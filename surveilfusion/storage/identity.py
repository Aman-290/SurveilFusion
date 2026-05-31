import json
import sqlite3
from pathlib import Path

from surveilfusion.core.models import FaceIdentity


class IdentityStore:
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
                CREATE TABLE IF NOT EXISTS identities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def add(self, identity: FaceIdentity) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO identities (id, name, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    identity.id,
                    identity.name,
                    identity.created_at.isoformat(),
                    identity.model_dump_json(),
                ),
            )

    def latest(self, limit: int = 100) -> list[FaceIdentity]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM identities ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [FaceIdentity.model_validate(json.loads(row["payload"])) for row in rows]
