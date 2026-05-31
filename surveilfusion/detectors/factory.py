from surveilfusion.core.config import Settings
from surveilfusion.detectors.base import Detector
from surveilfusion.detectors.yolo import YoloDetector


def build_detectors(settings: Settings) -> list[Detector]:
    detectors: list[Detector] = []
    if settings.fire_model_path.exists():
        detectors.append(YoloDetector(settings.fire_model_path, confidence=settings.default_confidence))
    return detectors
