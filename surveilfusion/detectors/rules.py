from collections.abc import Iterable

from surveilfusion.core.models import Detection, DetectionKind, EventSeverity, SurveillanceEvent

SEVERITY_BY_KIND = {
    DetectionKind.fire: EventSeverity.critical,
    DetectionKind.smoke: EventSeverity.high,
    DetectionKind.unknown_face: EventSeverity.medium,
    DetectionKind.distress_audio: EventSeverity.high,
    DetectionKind.person: EventSeverity.low,
    DetectionKind.motion: EventSeverity.info,
    DetectionKind.system: EventSeverity.info,
}


def detections_to_events(camera_id: str, detections: Iterable[Detection]) -> list[SurveillanceEvent]:
    events: list[SurveillanceEvent] = []
    for detection in detections:
        severity = SEVERITY_BY_KIND[detection.kind]
        events.append(
            SurveillanceEvent(
                camera_id=camera_id,
                kind=detection.kind,
                severity=severity,
                title=f"{detection.label.title()} detected",
                summary=f"{detection.label} detected with {detection.confidence:.0%} confidence.",
                detections=[detection],
            )
        )
    return events
