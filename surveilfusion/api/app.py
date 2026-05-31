from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from surveilfusion.agents.incident_agent import IncidentAgent
from surveilfusion.camera.registry import CameraRegistry
from surveilfusion.core.config import get_settings
from surveilfusion.core.models import Detection, DetectionKind, EventSeverity, SurveillanceEvent
from surveilfusion.memory.event_memory import EventMemory
from surveilfusion.storage.events import EventStore

settings = get_settings()
registry = CameraRegistry(settings.cameras_file)
store = EventStore(settings.data_dir / "surveilfusion.db")
agent = IncidentAgent()
memory = EventMemory()

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


@app.post("/api/events/{event_id}/ack")
async def acknowledge(event_id: str) -> dict[str, bool]:
    return {"acknowledged": store.acknowledge(event_id)}


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


def main() -> None:
    uvicorn.run(
        "surveilfusion.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
