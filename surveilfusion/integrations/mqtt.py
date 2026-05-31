import json
from dataclasses import dataclass

from surveilfusion.core.models import SurveillanceEvent


@dataclass(slots=True)
class MqttPayload:
    topic: str
    payload: str
    retain: bool = False


def event_to_mqtt(event: SurveillanceEvent, prefix: str = "surveilfusion") -> MqttPayload:
    return MqttPayload(
        topic=f"{prefix}/events/{event.camera_id}/{event.kind.value}",
        payload=json.dumps(
            {
                "id": event.id,
                "camera_id": event.camera_id,
                "kind": event.kind.value,
                "severity": event.severity.value,
                "title": event.title,
                "summary": event.summary,
                "created_at": event.created_at.isoformat(),
            }
        ),
    )
