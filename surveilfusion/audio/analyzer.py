import math
import wave
from pathlib import Path

from surveilfusion.core.models import (
    AudioAnalysisRun,
    Detection,
    DetectionKind,
    DetectionRunStatus,
)
from surveilfusion.detectors.rules import detections_to_events
from surveilfusion.storage.events import EventStore


class AudioAnalyzer:
    def __init__(
        self,
        event_store: EventStore,
        *,
        distress_dbfs_threshold: float = -18.0,
        min_duration_seconds: float = 0.25,
    ):
        self.event_store = event_store
        self.distress_dbfs_threshold = distress_dbfs_threshold
        self.min_duration_seconds = min_duration_seconds

    def analyze_wav(self, audio_path: Path, *, camera_id: str) -> AudioAnalysisRun:
        if not audio_path.exists():
            return AudioAnalysisRun(
                camera_id=camera_id,
                source=str(audio_path),
                status=DetectionRunStatus.failed,
                message="Audio path does not exist.",
            )

        try:
            metrics = wav_metrics(audio_path)
        except (wave.Error, OSError, EOFError) as exc:
            return AudioAnalysisRun(
                camera_id=camera_id,
                source=str(audio_path),
                status=DetectionRunStatus.failed,
                message=f"Could not read WAV audio: {exc}",
            )

        detections: list[Detection] = []
        if (
            metrics["duration_seconds"] >= self.min_duration_seconds
            and metrics["rms_dbfs"] >= self.distress_dbfs_threshold
        ):
            confidence = min(max((metrics["rms_dbfs"] - self.distress_dbfs_threshold) / 18 + 0.5, 0.5), 0.99)
            detections.append(
                Detection(
                    kind=DetectionKind.distress_audio,
                    label="loud_audio_anomaly",
                    confidence=round(confidence, 4),
                    metadata=metrics,
                )
            )

        events = detections_to_events(camera_id, detections)
        for event in events:
            self.event_store.add(event)

        return AudioAnalysisRun(
            camera_id=camera_id,
            source=str(audio_path),
            status=DetectionRunStatus.completed,
            duration_seconds=metrics["duration_seconds"],
            rms_dbfs=metrics["rms_dbfs"],
            peak=metrics["peak"],
            zero_crossing_rate=metrics["zero_crossing_rate"],
            detections=detections,
            events=events,
            message=f"Created {len(events)} events from audio analysis.",
        )


def wav_metrics(audio_path: Path) -> dict[str, float]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    samples = _pcm_samples(frames, sample_width)
    max_possible = float(2 ** (8 * sample_width - 1))
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1))
    peak = max((abs(sample) for sample in samples), default=0)
    rms_dbfs = 20 * math.log10(max(rms / max_possible, 1e-9))
    zero_crossing_rate = _zero_crossing_rate(samples)

    return {
        "duration_seconds": round(frame_count / frame_rate, 4) if frame_rate else 0,
        "rms_dbfs": round(rms_dbfs, 4),
        "peak": round(peak / max_possible, 4),
        "zero_crossing_rate": round(zero_crossing_rate, 4),
        "channels": float(channels),
        "sample_rate": float(frame_rate),
    }


def _pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    if sample_width not in {1, 2, 3, 4}:
        raise wave.Error(f"Unsupported PCM sample width: {sample_width}")

    samples: list[int] = []
    for start in range(0, len(frames), sample_width):
        sample_bytes = frames[start : start + sample_width]
        if len(sample_bytes) != sample_width:
            continue
        if sample_width == 1:
            samples.append(sample_bytes[0] - 128)
        else:
            samples.append(int.from_bytes(sample_bytes, byteorder="little", signed=True))
    return samples


def _zero_crossing_rate(samples: list[int]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous < 0 <= sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    return crossings / (len(samples) - 1)
