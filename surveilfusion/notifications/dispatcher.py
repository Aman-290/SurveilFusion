from datetime import datetime, timezone

import httpx

from surveilfusion.core.config import Settings
from surveilfusion.core.models import NotificationChannel, NotificationMessage, NotificationStatus


class NotificationDispatcher:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def dispatch(self, notification: NotificationMessage) -> NotificationMessage:
        if notification.channel == NotificationChannel.outbox:
            notification.status = NotificationStatus.skipped
            notification.error = "Local outbox entries are not sent externally."
            return notification

        if notification.channel == NotificationChannel.webhook:
            return await self._dispatch_webhook(notification)

        notification.status = NotificationStatus.skipped
        notification.error = f"{notification.channel.value} dispatch is not configured yet."
        return notification

    async def _dispatch_webhook(self, notification: NotificationMessage) -> NotificationMessage:
        if not notification.target:
            notification.status = NotificationStatus.failed
            notification.error = "Webhook notification missing target URL."
            return notification
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(notification.target, json=notification.payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            notification.status = NotificationStatus.failed
            notification.error = str(exc)
            return notification

        notification.status = NotificationStatus.sent
        notification.sent_at = datetime.now(timezone.utc)
        return notification
