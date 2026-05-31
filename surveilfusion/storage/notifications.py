import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from surveilfusion.core.models import NotificationMessage, NotificationStatus


class NotificationStore:
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
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def add(self, notification: NotificationMessage) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO notifications
                (id, event_id, camera_id, channel, status, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification.id,
                    notification.event_id,
                    notification.camera_id,
                    notification.channel.value,
                    notification.status.value,
                    notification.created_at.isoformat(),
                    notification.model_dump_json(),
                ),
            )

    def get(self, notification_id: str) -> NotificationMessage | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM notifications WHERE id = ?", (notification_id,)).fetchone()
        if row is None:
            return None
        return NotificationMessage.model_validate(json.loads(row["payload"]))

    def latest(self, limit: int = 50) -> list[NotificationMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM notifications ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [NotificationMessage.model_validate(json.loads(row["payload"])) for row in rows]

    def mark_sent(self, notification_id: str) -> NotificationMessage | None:
        notification = self.get(notification_id)
        if notification is None:
            return None
        notification.status = NotificationStatus.sent
        notification.sent_at = datetime.now(timezone.utc)
        self.add(notification)
        return notification
