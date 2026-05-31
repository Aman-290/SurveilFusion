import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from surveilfusion.core.models import ActionRequest, ActionStatus


class ActionStore:
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
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    event_id TEXT,
                    status TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def add(self, action: ActionRequest) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO actions
                (id, kind, camera_id, event_id, status, risk, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.id,
                    action.kind.value,
                    action.camera_id,
                    action.event_id,
                    action.status.value,
                    action.risk.value,
                    action.created_at.isoformat(),
                    action.model_dump_json(),
                ),
            )

    def get(self, action_id: str) -> ActionRequest | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM actions WHERE id = ?", (action_id,)).fetchone()
        if row is None:
            return None
        return ActionRequest.model_validate(json.loads(row["payload"]))

    def latest(self, limit: int = 50) -> list[ActionRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM actions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ActionRequest.model_validate(json.loads(row["payload"])) for row in rows]

    def approve(self, action_id: str) -> ActionRequest | None:
        action = self.get(action_id)
        if action is None:
            return None
        action.status = ActionStatus.approved
        action.approved_at = datetime.now(timezone.utc)
        self.add(action)
        return action

    def deny(self, action_id: str, reason: str = "Denied by operator.") -> ActionRequest | None:
        action = self.get(action_id)
        if action is None:
            return None
        action.status = ActionStatus.denied
        action.result = reason
        self.add(action)
        return action
