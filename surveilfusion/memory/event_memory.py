import math
import re
from collections import Counter
from datetime import datetime, timezone

from surveilfusion.core.models import EventSeverity, MemorySearchResult, SurveillanceEvent


class EventMemory:
    """Local retrieval layer for incident memory.

    This is intentionally dependency-free so the app is useful immediately after clone.
    Qdrant-backed vector search can later plug in behind the same API.
    """

    def summarize(self, events: list[SurveillanceEvent]) -> dict[str, object]:
        by_kind = Counter(event.kind.value for event in events)
        by_camera = Counter(event.camera_id for event in events)
        by_severity = Counter(event.severity.value for event in events)
        return {
            "total_events": len(events),
            "by_kind": dict(by_kind),
            "by_camera": dict(by_camera),
            "by_severity": dict(by_severity),
            "unacknowledged": sum(1 for event in events if not event.acknowledged),
        }

    def search(
        self,
        events: list[SurveillanceEvent],
        query: str,
        *,
        limit: int = 10,
        camera_id: str | None = None,
        kind: str | None = None,
    ) -> list[MemorySearchResult]:
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        results: list[MemorySearchResult] = []
        now = datetime.now(timezone.utc)
        for event in events:
            if camera_id and event.camera_id != camera_id:
                continue
            if kind and event.kind.value != kind:
                continue

            document_terms = set(_tokenize(_event_text(event)))
            matched_terms = sorted(query_terms & document_terms)
            if not matched_terms:
                continue

            text_score = len(matched_terms) / max(len(query_terms), 1)
            severity_boost = SEVERITY_WEIGHT[event.severity]
            recency_boost = _recency_boost(now, event.created_at)
            score = round(text_score * 0.72 + severity_boost * 0.18 + recency_boost * 0.10, 4)
            results.append(
                MemorySearchResult(
                    event=event,
                    score=score,
                    matched_terms=matched_terms,
                    rationale=(
                        f"Matched {', '.join(matched_terms)} with severity {event.severity.value} "
                        f"and recency boost {recency_boost:.2f}."
                    ),
                )
            )

        results.sort(key=lambda result: (result.score, result.event.created_at), reverse=True)
        return results[:limit]

    def similar(self, events: list[SurveillanceEvent], event_id: str, *, limit: int = 5) -> list[MemorySearchResult]:
        anchor = next((event for event in events if event.id == event_id), None)
        if anchor is None:
            return []
        query = f"{anchor.kind.value} {anchor.camera_id} {anchor.title} {anchor.summary}"
        candidates = [event for event in events if event.id != event_id]
        return self.search(candidates, query, limit=limit, camera_id=anchor.camera_id)


SEVERITY_WEIGHT = {
    EventSeverity.info: 0.05,
    EventSeverity.low: 0.10,
    EventSeverity.medium: 0.25,
    EventSeverity.high: 0.55,
    EventSeverity.critical: 0.85,
}


def _event_text(event: SurveillanceEvent) -> str:
    detection_text = " ".join(f"{detection.kind.value} {detection.label}" for detection in event.detections)
    metadata_text = " ".join(str(value) for value in event.metadata.values())
    return " ".join(
        [
            event.id,
            event.camera_id,
            event.kind.value,
            event.severity.value,
            event.title,
            event.summary,
            detection_text,
            metadata_text,
        ]
    )


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_]+", value.lower()) if token not in STOP_WORDS]


def _recency_boost(now: datetime, created_at: datetime) -> float:
    age_hours = max((now - created_at).total_seconds() / 3600, 0)
    return math.exp(-age_hours / 168)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}
