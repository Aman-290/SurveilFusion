from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from surveilfusion.actions.executor import ActionExecutor
from surveilfusion.actions.policy import ActionPolicy
from surveilfusion.agents.incident_agent import IncidentAgent
from surveilfusion.camera.go2rtc import generate_frigate_camera_block, generate_go2rtc_config
from surveilfusion.camera.probe import probe_camera
from surveilfusion.camera.registry import CameraRegistry
from surveilfusion.core.config import get_settings
from surveilfusion.core.models import (
    ActionCreate,
    ActionRequest,
    Detection,
    DetectionKind,
    EventSeverity,
    SurveillanceEvent,
)
from surveilfusion.integrations.home_assistant import discovery_messages
from surveilfusion.memory.event_memory import EventMemory
from surveilfusion.storage.actions import ActionStore
from surveilfusion.storage.events import EventStore

settings = get_settings()
registry = CameraRegistry(settings.cameras_file)
store = EventStore(settings.data_dir / "surveilfusion.db")
agent = IncidentAgent()
memory = EventMemory()
action_store = ActionStore(settings.data_dir / "surveilfusion.db")
action_policy = ActionPolicy()
action_executor = ActionExecutor()

app = FastAPI(
    title="SurveilFusion",
    version="0.2.0",
    description="Local-first AI surveillance, CCTV integration, incident memory, and agentic automation.",
)

static_dir = Path("web/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return Path("web/static/index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/api/cameras")
async def cameras() -> list[dict]:
    return [state.model_dump(mode="json") for state in registry.all()]


@app.get("/api/cameras/{camera_id}/probe")
async def camera_probe(camera_id: str) -> dict:
    state = registry.get(camera_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return probe_camera(state.camera.source).__dict__


@app.get("/api/integrations/go2rtc")
async def go2rtc_config() -> dict:
    cameras_config = [state.camera for state in registry.all()]
    return generate_go2rtc_config(cameras_config)


@app.get("/api/integrations/frigate")
async def frigate_config() -> dict:
    cameras_config = [state.camera for state in registry.all()]
    return {"cameras": generate_frigate_camera_block(cameras_config)}


@app.get("/api/integrations/home-assistant/mqtt-discovery")
async def home_assistant_discovery() -> list[dict[str, object]]:
    cameras_config = [state.camera for state in registry.all()]
    return discovery_messages(cameras_config)


@app.get("/api/events")
async def events(limit: int = 50) -> list[dict]:
    return [event.model_dump(mode="json") for event in store.latest(limit=limit)]


@app.post("/api/events/demo")
async def create_demo_event(camera_id: str = "front-door", kind: DetectionKind = DetectionKind.fire) -> dict:
    event = SurveillanceEvent(
        camera_id=camera_id,
        kind=kind,
        severity=EventSeverity.critical if kind == DetectionKind.fire else EventSeverity.medium,
        title=f"Demo {kind.value.replace('_', ' ')} event",
        summary="Synthetic event generated to verify the API, dashboard, memory, and agent pipeline.",
        detections=[Detection(kind=kind, label=kind.value, confidence=0.91)],
    )
    store.add(event)
    return event.model_dump(mode="json")


@app.get("/api/events/{event_id}/recommendation")
async def recommendation(event_id: str) -> dict:
    for event in store.latest(limit=250):
        if event.id == event_id:
            return agent.recommend(event).model_dump(mode="json")
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/events/{event_id}/actions/propose")
async def propose_event_actions(event_id: str) -> list[dict]:
    for event in store.latest(limit=250):
        if event.id == event_id:
            actions = []
            for proposal in agent.propose_actions(event):
                action = _create_action(proposal)
                actions.append(action.model_dump(mode="json"))
            return actions
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/events/{event_id}/ack")
async def acknowledge(event_id: str) -> dict[str, bool]:
    return {"acknowledged": store.acknowledge(event_id)}


@app.get("/api/actions")
async def actions(limit: int = 50) -> list[dict]:
    return [action.model_dump(mode="json") for action in action_store.latest(limit=limit)]


@app.post("/api/actions")
async def create_action(action: ActionCreate) -> dict:
    return _create_action(action).model_dump(mode="json")


@app.post("/api/actions/{action_id}/approve")
async def approve_action(action_id: str) -> dict:
    action = action_store.approve(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    return action.model_dump(mode="json")


@app.post("/api/actions/{action_id}/deny")
async def deny_action(action_id: str, reason: str = "Denied by operator.") -> dict:
    action = action_store.deny(action_id, reason=reason)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    return action.model_dump(mode="json")


@app.post("/api/actions/{action_id}/execute")
async def execute_action(action_id: str) -> dict:
    action = action_store.get(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    action = action_executor.execute(action)
    action_store.add(action)
    return action.model_dump(mode="json")


@app.get("/api/memory/summary")
async def memory_summary() -> dict[str, object]:
    return memory.summarize(store.latest(limit=500))


@app.websocket("/ws/events")
async def events_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await websocket.send_json({"type": "hello", "message": "SurveilFusion event stream connected"})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return


def _create_action(action: ActionCreate) -> ActionRequest:
    decision = action_policy.evaluate(action)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)
    action_request = ActionRequest(
        kind=action.kind,
        camera_id=action.camera_id,
        event_id=action.event_id,
        reason=action.reason,
        requested_by=action.requested_by,
        risk=decision.risk,
        requires_approval=decision.requires_approval,
        metadata={**action.metadata, "policy_reason": decision.reason},
    )
    if not action_request.requires_approval:
        action_request = action_executor.execute(action_request)
    action_store.add(action_request)
    return action_request


def main() -> None:
    uvicorn.run(
        "surveilfusion.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
