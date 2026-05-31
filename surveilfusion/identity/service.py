import math

from surveilfusion.core.models import (
    Detection,
    DetectionKind,
    EventSeverity,
    FaceEnrollment,
    FaceIdentificationRequest,
    FaceIdentificationResult,
    FaceIdentity,
    SurveillanceEvent,
)
from surveilfusion.storage.events import EventStore
from surveilfusion.storage.identity import IdentityStore


class FaceIdentityService:
    def __init__(self, identity_store: IdentityStore, event_store: EventStore):
        self.identity_store = identity_store
        self.event_store = event_store

    def enroll(self, enrollment: FaceEnrollment) -> FaceIdentity:
        identity = FaceIdentity(
            name=enrollment.name,
            embedding=_normalize(enrollment.embedding),
            source=enrollment.source,
            metadata=enrollment.metadata,
        )
        self.identity_store.add(identity)
        return identity

    def identify(self, request: FaceIdentificationRequest) -> FaceIdentificationResult:
        probe = _normalize(request.embedding)
        best_identity: FaceIdentity | None = None
        best_score = 0.0

        for identity in self.identity_store.latest(limit=500):
            score = cosine_similarity(probe, identity.embedding)
            if score > best_score:
                best_identity = identity
                best_score = score

        if best_identity and best_score >= request.threshold:
            return FaceIdentificationResult(
                camera_id=request.camera_id,
                matched=True,
                identity=best_identity,
                score=round(best_score, 4),
                threshold=request.threshold,
            )

        event = SurveillanceEvent(
            camera_id=request.camera_id,
            kind=DetectionKind.unknown_face,
            severity=EventSeverity.medium,
            title="Unknown face detected",
            summary=f"Unknown face observed on {request.camera_id}.",
            detections=[
                Detection(
                    kind=DetectionKind.unknown_face,
                    label="unknown_face",
                    confidence=round(max(1 - best_score, 0.01), 4),
                    metadata={"source": request.source, "best_score": round(best_score, 4)},
                )
            ],
            metadata={"source": request.source, "best_score": round(best_score, 4)},
        )
        self.event_store.add(event)
        return FaceIdentificationResult(
            camera_id=request.camera_id,
            matched=False,
            identity=best_identity,
            score=round(best_score, 4),
            threshold=request.threshold,
            event=event,
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (_norm(left) * _norm(right) or 1)


def _normalize(values: list[float]) -> list[float]:
    norm = _norm(values)
    if norm == 0:
        return values
    return [value / norm for value in values]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
