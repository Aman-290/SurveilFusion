from pathlib import Path

import yaml

from surveilfusion.core.models import CameraConfig, CameraState, CameraStatus


class CameraRegistry:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._states: dict[str, CameraState] = {}

    def load(self) -> list[CameraState]:
        if not self.config_path.exists():
            self._states = {}
            return []

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        cameras = [CameraConfig(**item) for item in payload.get("cameras", [])]
        self._states = {
            camera.id: CameraState(camera=camera, status=CameraStatus.offline)
            for camera in cameras
            if camera.enabled
        }
        return list(self._states.values())

    def all(self) -> list[CameraState]:
        if not self._states:
            self.load()
        return list(self._states.values())

    def get(self, camera_id: str) -> CameraState | None:
        if not self._states:
            self.load()
        return self._states.get(camera_id)
