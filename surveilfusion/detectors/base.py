from abc import ABC, abstractmethod
from typing import Any

from surveilfusion.core.models import Detection


class Detector(ABC):
    name: str

    @abstractmethod
    async def detect(self, frame: Any, *, camera_id: str) -> list[Detection]:
        """Return detections for one decoded video frame."""
