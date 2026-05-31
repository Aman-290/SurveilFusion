import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from surveilfusion.actions.executor import ActionExecutor
from surveilfusion.actions.policy import ActionPolicy
from surveilfusion.agents.incident_agent import IncidentAgent
from surveilfusion.camera.go2rtc import generate_frigate_camera_block, generate_go2rtc_config
from surveilfusion.core.models import (
    ActionCreate,
    ActionKind,
    ActionStatus,
    CameraConfig,
    Detection,
    DetectionKind,
    EventSeverity,
    SurveillanceEvent,
)
from surveilfusion.detectors.service import DetectionService
from surveilfusion.integrations.home_assistant import discovery_messages
from surveilfusion.integrations.mqtt import event_to_mqtt
from surveilfusion.memory.event_memory import EventMemory
from surveilfusion.onboarding import export_integration_configs, initialize_project, run_doctor
from surveilfusion.security import is_authorized, is_public_path
from surveilfusion.storage.actions import ActionStore
from surveilfusion.storage.events import EventStore


class FakeFireDetector:
    name = "fake-fire"

    async def detect(self, frame, *, camera_id: str):
        return [Detection(kind=DetectionKind.fire, label="fire", confidence=0.94)]


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


def test_detection_service_creates_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    service = DetectionService([FakeFireDetector()], store)

    missing_run = service.run_on_image_path_sync(tmp_path / "missing.jpg", camera_id="front-door")
    frame_run = asyncio.run(service.run_on_frame(object(), camera_id="front-door", source="unit-test"))

    assert missing_run.status.value == "failed"
    assert frame_run.events[0].severity == EventSeverity.critical
    assert store.latest()[0].kind == DetectionKind.fire


def test_api_key_security_helpers() -> None:
    assert is_public_path("/")
    assert is_public_path("/static/app.js")
    assert not is_public_path("/api/actions")
    assert is_authorized({"x-surveilfusion-key": "secret"}, "secret")
    assert is_authorized({"authorization": "Bearer secret"}, "secret")
    assert not is_authorized({"authorization": "Bearer wrong"}, "secret")


def test_agent_and_memory_outputs() -> None:
    event = SurveillanceEvent(
        camera_id="driveway",
        kind=DetectionKind.unknown_face,
        severity=EventSeverity.medium,
        title="Unknown face detected",
        summary="Unknown person at driveway.",
    )

    recommendation = IncidentAgent().recommend(event)
    proposed_actions = IncidentAgent().propose_actions(event)
    summary = EventMemory().summarize([event])
    mqtt_payload = event_to_mqtt(event)

    assert recommendation.priority == EventSeverity.medium
    assert [action.kind for action in proposed_actions] == [ActionKind.notify, ActionKind.pin_live_view]
    assert summary["unacknowledged"] == 1
    assert mqtt_payload.topic == "surveilfusion/events/driveway/unknown_face"


def test_memory_search_and_similarity() -> None:
    memory = EventMemory()
    fire_event = SurveillanceEvent(
        camera_id="front-door",
        kind=DetectionKind.fire,
        severity=EventSeverity.critical,
        title="Fire detected at front door",
        summary="Flames near the front entry.",
        created_at=datetime.now(timezone.utc),
    )
    old_face_event = SurveillanceEvent(
        camera_id="front-door",
        kind=DetectionKind.unknown_face,
        severity=EventSeverity.medium,
        title="Unknown face at front door",
        summary="Visitor near the entry.",
        created_at=datetime.now(timezone.utc) - timedelta(days=10),
    )
    driveway_event = SurveillanceEvent(
        camera_id="driveway",
        kind=DetectionKind.fire,
        severity=EventSeverity.high,
        title="Smoke in driveway",
        summary="Smoke plume near parked car.",
    )

    results = memory.search([old_face_event, driveway_event, fire_event], "fire front door")
    similar = memory.similar([old_face_event, driveway_event, fire_event], fire_event.id)

    assert results[0].event.id == fire_event.id
    assert "front" in results[0].matched_terms
    assert similar[0].event.id == old_face_event.id


def test_action_policy_store_and_executor(tmp_path: Path) -> None:
    store = ActionStore(tmp_path / "actions.db")
    policy = ActionPolicy()
    executor = ActionExecutor()

    low_risk = ActionCreate(kind=ActionKind.notify, camera_id="front-door", reason="Alert owner.")
    decision = policy.evaluate(low_risk)
    low_risk_action = build_action_request(low_risk, decision.requires_approval, decision.risk)
    action = executor.execute(low_risk_action)
    store.add(action)

    high_risk = ActionCreate(
        kind=ActionKind.open_two_way_audio,
        camera_id="front-door",
        reason="Operator wants to speak through the camera.",
    )
    high_decision = policy.evaluate(high_risk)
    high_action = build_action_request(high_risk, high_decision.requires_approval, high_decision.risk)
    store.add(high_action)
    approved = store.approve(high_action.id)

    assert low_risk_action.requires_approval is False
    assert action.status == ActionStatus.executed
    assert high_decision.requires_approval is True
    assert approved is not None
    assert executor.execute(approved).status == ActionStatus.executed


def build_action_request(action: ActionCreate, requires_approval: bool, risk):
    from surveilfusion.core.models import ActionRequest

    return ActionRequest(
        kind=action.kind,
        camera_id=action.camera_id,
        reason=action.reason,
        requires_approval=requires_approval,
        risk=risk,
    )


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


def test_onboarding_init_doctor_and_export(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / ".env.example").write_text("PORT=8080\n", encoding="utf-8")
    (tmp_path / "config" / "cameras.example.yml").write_text(
        """
cameras:
  - id: front-door
    name: Front Door
    source: rtsp://user:pass@camera/stream1
""",
        encoding="utf-8",
    )

    created = initialize_project(tmp_path)
    report = run_doctor(tmp_path)
    exported = export_integration_configs(tmp_path / "config" / "cameras.yml", tmp_path / "generated")

    assert ".env" in created
    assert "config\\cameras.yml" in created or "config/cameras.yml" in created
    assert report["checks"]["env_file"] is True
    assert {path.name for path in exported} == {
        "go2rtc.yml",
        "frigate.cameras.yml",
        "home-assistant-mqtt-discovery.json",
    }
