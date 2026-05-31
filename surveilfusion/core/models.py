from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class CameraStatus(str, Enum):
    offline = "offline"
    connecting = "connecting"
    online = "online"
    degraded = "degraded"


class DetectionKind(str, Enum):
    fire = "fire"
    smoke = "smoke"
    person = "person"
    unknown_face = "unknown_face"
    distress_audio = "distress_audio"
    motion = "motion"
    system = "system"


class CameraConfig(BaseModel):
    id: str
    name: str
    source: str
    zone: str = "default"
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    detect_fps: float = Field(default=3.0, gt=0, le=30)
    record: bool = False
    ptz: bool = False
    audio: bool = False


class CameraState(BaseModel):
    camera: CameraConfig
    status: CameraStatus = CameraStatus.offline
    last_seen_at: datetime | None = None
    health: dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    kind: DetectionKind
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventSeverity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class SurveillanceEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    camera_id: str
    kind: DetectionKind
    severity: EventSeverity
    title: str
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detections: list[Detection] = Field(default_factory=list)
    snapshot_path: str | None = None
    clip_path: str | None = None
    acknowledged: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRecommendation(BaseModel):
    event_id: str
    priority: EventSeverity
    rationale: str
    recommended_actions: list[str]
    automation_plan: list[str] = Field(default_factory=list)


class ActionKind(str, Enum):
    notify = "notify"
    pin_live_view = "pin_live_view"
    start_recording = "start_recording"
    publish_mqtt = "publish_mqtt"
    trigger_home_assistant_scene = "trigger_home_assistant_scene"
    move_ptz_preset = "move_ptz_preset"
    open_two_way_audio = "open_two_way_audio"


class ActionStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    denied = "denied"
    executed = "executed"
    failed = "failed"


class ActionRisk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActionRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: ActionKind
    camera_id: str
    event_id: str | None = None
    reason: str
    requested_by: str = "agent"
    status: ActionStatus = ActionStatus.proposed
    risk: ActionRisk = ActionRisk.low
    requires_approval: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: datetime | None = None
    executed_at: datetime | None = None
    result: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionCreate(BaseModel):
    kind: ActionKind
    camera_id: str
    event_id: str | None = None
    reason: str
    requested_by: str = "agent"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionPolicyDecision(BaseModel):
    allowed: bool
    risk: ActionRisk
    requires_approval: bool
    reason: str
