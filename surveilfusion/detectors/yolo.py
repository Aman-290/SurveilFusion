from pathlib import Path
from typing import Any

from surveilfusion.core.models import Detection, DetectionKind
from surveilfusion.detectors.base import Detector


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, model_path: Path, confidence: float = 0.55):
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    async def detect(self, frame: Any, *, camera_id: str) -> list[Detection]:
        if not self.model_path.exists():
            return []

        model = self._load()
        results = model(frame, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence:
                    continue
                label = names[int(box.cls[0])]
                kind = DetectionKind.fire if "fire" in label.lower() else DetectionKind.person
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        kind=kind,
                        label=label,
                        confidence=confidence,
                        box=xyxy,
                        metadata={"camera_id": camera_id, "model": self.model_path.name},
                    )
                )
        return detections
