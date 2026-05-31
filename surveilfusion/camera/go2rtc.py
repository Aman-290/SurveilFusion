from surveilfusion.core.models import CameraConfig


def generate_go2rtc_config(cameras: list[CameraConfig]) -> dict[str, dict[str, list[str]]]:
    streams: dict[str, list[str]] = {}
    for camera in cameras:
        streams[camera.id] = [camera.source]
        if camera.audio:
            streams[f"{camera.id}_twoway"] = [camera.source.replace("#backchannel=0", "")]
    return {"streams": streams}


def generate_frigate_camera_block(cameras: list[CameraConfig]) -> dict[str, dict]:
    return {
        camera.id: {
            "ffmpeg": {
                "inputs": [
                    {
                        "path": f"rtsp://127.0.0.1:8554/{camera.id}",
                        "input_args": "preset-rtsp-restream",
                        "roles": ["detect", "record"] if camera.record else ["detect"],
                    }
                ]
            },
            "detect": {"fps": camera.detect_fps},
            "record": {"enabled": camera.record},
        }
        for camera in cameras
    }
