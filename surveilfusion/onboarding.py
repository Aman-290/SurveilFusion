import json
import shutil
import subprocess
from pathlib import Path

import yaml

from surveilfusion.camera.go2rtc import generate_frigate_camera_block, generate_go2rtc_config
from surveilfusion.camera.registry import CameraRegistry
from surveilfusion.integrations.home_assistant import discovery_messages


def copy_template(source: Path, destination: Path, *, force: bool = False) -> bool:
    if destination.exists() and not force:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def initialize_project(root: Path, *, force: bool = False) -> list[str]:
    created: list[str] = []
    templates = [
        (root / ".env.example", root / ".env"),
        (root / "config" / "cameras.example.yml", root / "config" / "cameras.yml"),
    ]
    for source, destination in templates:
        if copy_template(source, destination, force=force):
            created.append(str(destination.relative_to(root)))
    return created


def run_doctor(root: Path) -> dict[str, object]:
    checks = {
        "env_file": (root / ".env").exists(),
        "camera_config": (root / "config" / "cameras.yml").exists()
        or (root / "config" / "cameras.example.yml").exists(),
        "docker": _command_ok(["docker", "--version"]),
        "docker_compose": _command_ok(["docker", "compose", "version"]),
        "data_dir_writable": _can_write(root / "data"),
        "fire_model": (root / "models" / "fire-yolo.pt").exists(),
    }
    return {
        "ok": all(value for name, value in checks.items() if name != "fire_model"),
        "checks": checks,
        "next_steps": _next_steps(checks),
    }


def export_integration_configs(cameras_file: Path, output_dir: Path) -> list[Path]:
    registry = CameraRegistry(cameras_file)
    cameras = [state.camera for state in registry.load()]
    output_dir.mkdir(parents=True, exist_ok=True)

    go2rtc_path = output_dir / "go2rtc.yml"
    frigate_path = output_dir / "frigate.cameras.yml"
    home_assistant_path = output_dir / "home-assistant-mqtt-discovery.json"

    go2rtc_path.write_text(yaml.safe_dump(generate_go2rtc_config(cameras), sort_keys=False), encoding="utf-8")
    frigate_path.write_text(
        yaml.safe_dump({"cameras": generate_frigate_camera_block(cameras)}, sort_keys=False),
        encoding="utf-8",
    )
    home_assistant_path.write_text(
        json.dumps(discovery_messages(cameras), indent=2),
        encoding="utf-8",
    )

    return [go2rtc_path, frigate_path, home_assistant_path]


def _can_write(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _command_ok(command: list[str]) -> bool:
    try:
        return subprocess.run(command, capture_output=True, check=False, text=True).returncode == 0
    except OSError:
        return False


def _next_steps(checks: dict[str, bool]) -> list[str]:
    steps: list[str] = []
    if not checks["env_file"]:
        steps.append("Run `surveilfusion init` to create .env.")
    if not checks["camera_config"]:
        steps.append("Create config/cameras.yml with your RTSP or ONVIF camera sources.")
    if not checks["docker"]:
        steps.append("Install Docker Desktop for the one-command local stack.")
    if not checks["fire_model"]:
        steps.append("Optional: add models/fire-yolo.pt or set FIRE_MODEL_PATH to enable image detection.")
    if not steps:
        steps.append("Run `docker compose up --build` or `python -m surveilfusion`.")
    return steps
