from pathlib import Path

from surveilfusion.agents.incident_agent import IncidentAgent
from surveilfusion.camera.go2rtc import generate_frigate_camera_block, generate_go2rtc_config
from surveilfusion.core.models import CameraConfig, Detection, DetectionKind, EventSeverity, SurveillanceEvent
from surveilfusion.integrations.home_assistant import discovery_messages
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


def test_camera_integration_configs() -> None:
    camera = CameraConfig(
        id="front-door",
        name="Front Door",
        source="rtsp://user:pass@camera/stream1#backchannel=0",
        audio=True,
        record=True,
    )

    go2rtc = generate_go2rtc_config([camera])
    frigate = generate_frigate_camera_block([camera])
    discovery = discovery_messages([camera])

    assert go2rtc["streams"]["front-door"][0].startswith("rtsp://")
    assert "front-door_twoway" in go2rtc["streams"]
    assert frigate["front-door"]["record"]["enabled"] is True
    assert discovery[0]["topic"] == "homeassistant/binary_sensor/surveilfusion/front-door/config"
