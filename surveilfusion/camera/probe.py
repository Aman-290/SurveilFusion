from dataclasses import dataclass


@dataclass(slots=True)
class CameraProbeResult:
    ok: bool
    message: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def probe_camera(source: str, timeout_seconds: float = 5.0) -> CameraProbeResult:
    try:
        import cv2
    except ImportError:
        return CameraProbeResult(
            ok=False,
            message="Install the vision extra to probe cameras: pip install -e '.[vision]'",
        )

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_seconds * 1000))
    ok, _frame = capture.read()
    if not ok:
        capture.release()
        return CameraProbeResult(ok=False, message="Could not read a frame from the camera source.")

    result = CameraProbeResult(
        ok=True,
        message="Camera source is readable.",
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
        fps=float(capture.get(cv2.CAP_PROP_FPS)) or None,
    )
    capture.release()
    return result
