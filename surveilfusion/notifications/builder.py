from surveilfusion.core.models import NotificationChannel, NotificationMessage, SurveillanceEvent


class NotificationBuilder:
    def for_event(
        self,
        event: SurveillanceEvent,
        *,
        channels: list[NotificationChannel] | None = None,
    ) -> list[NotificationMessage]:
        selected = channels or [NotificationChannel.outbox]
        return [
            NotificationMessage(
                event_id=event.id,
                camera_id=event.camera_id,
                channel=channel,
                title=f"{event.severity.value.upper()}: {event.title}",
                body=_event_body(event),
                target=_default_target(channel),
                payload={
                    "event": event.model_dump(mode="json"),
                    "severity": event.severity.value,
                    "kind": event.kind.value,
                    "camera_id": event.camera_id,
                },
            )
            for channel in selected
        ]


def _event_body(event: SurveillanceEvent) -> str:
    return f"{event.summary} Camera: {event.camera_id}. Event: {event.id}."


def _default_target(channel: NotificationChannel) -> str | None:
    if channel == NotificationChannel.outbox:
        return "local-outbox"
    return None
