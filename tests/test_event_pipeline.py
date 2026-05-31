from pathlib import Path

from surveilfusion.agents.incident_agent import IncidentAgent
from surveilfusion.core.models import Detection, DetectionKind, EventSeverity, SurveillanceEvent
from surveilfusion.integrations.mqtt import event_to_mqtt
from surveilfusion.memory.event_memory import EventMemory
from surveilfusion.storage.events import EventStore


def test_event_store_round_trip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = SurveillanceEvent(
        camera_id="front-door",
        kind=DetectionKind.fire,
        severity=EventSeverity.critical,
        title="Fire detected",
        summary="Fire detected with high confidence.",
        detections=[Detection(kind=DetectionKind.fire, label="fire", confidence=0.93)],
    )

    store.add(event)
    latest = store.latest()

    assert latest[0].id == event.id
    assert latest[0].detections[0].label == "fire"


def test_agent_and_memory_outputs() -> None:
    event = SurveillanceEvent(
        camera_id="driveway",
        kind=DetectionKind.unknown_face,
        severity=EventSeverity.medium,
        title="Unknown face detected",
        summary="Unknown person at driveway.",
    )

    recommendation = IncidentAgent().recommend(event)
    summary = EventMemory().summarize([event])
    mqtt_payload = event_to_mqtt(event)

    assert recommendation.priority == EventSeverity.medium
    assert summary["unacknowledged"] == 1
    assert mqtt_payload.topic == "surveilfusion/events/driveway/unknown_face"
