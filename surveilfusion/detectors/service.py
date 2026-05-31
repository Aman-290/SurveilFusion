import asyncio
from pathlib import Path
from typing import Any

from surveilfusion.core.models import DetectionRun, DetectionRunStatus
from surveilfusion.detectors.base import Detector
from surveilfusion.detectors.rules import detections_to_events
from surveilfusion.storage.events import EventStore


class DetectionService:
    def __init__(self, detectors: list[Detector], event_store: EventStore):
        self.detectors = detectors
        self.event_store = event_store

    async def run_on_frame(self, frame: Any, *, camera_id: str, source: str) -> DetectionRun:
        if not self.detectors:
            return DetectionRun(
                camera_id=camera_id,
                source=source,
                status=DetectionRunStatus.skipped,
                message="No detectors are configured.",
            )

        detections = []
        for detector in self.detectors:
            detections.extend(await detector.detect(frame, camera_id=camera_id))

        events = detections_to_events(camera_id, detections)
        for event in events:
            self.event_store.add(event)

        return DetectionRun(
            camera_id=camera_id,
            source=source,
            status=DetectionRunStatus.completed,
            detections=detections,
            events=events,
            message=f"Created {len(events)} events from {len(detections)} detections.",
        )

    async def run_on_image_path(self, image_path: Path, *, camera_id: str) -> DetectionRun:
        if not image_path.exists():
            return DetectionRun(
                camera_id=camera_id,
                source=str(image_path),
                status=DetectionRunStatus.failed,
                message="Image path does not exist.",
            )
        frame = _read_image(image_path)
        if frame is None:
            return DetectionRun(
                camera_id=camera_id,
                source=str(image_path),
                status=DetectionRunStatus.failed,
                message="Install the vision extra or provide a readable image file.",
            )
        return await self.run_on_frame(frame, camera_id=camera_id, source=str(image_path))

    def run_on_image_path_sync(self, image_path: Path, *, camera_id: str) -> DetectionRun:
        return asyncio.run(self.run_on_image_path(image_path, camera_id=camera_id))


def _read_image(image_path: Path) -> Any | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2.imread(str(image_path))
