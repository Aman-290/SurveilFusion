from collections import Counter

from surveilfusion.core.models import SurveillanceEvent


class EventMemory:
    """Small local memory layer for event summaries and future vector-store integration."""

    def summarize(self, events: list[SurveillanceEvent]) -> dict[str, object]:
        by_kind = Counter(event.kind.value for event in events)
        by_camera = Counter(event.camera_id for event in events)
        return {
            "total_events": len(events),
            "by_kind": dict(by_kind),
            "by_camera": dict(by_camera),
            "unacknowledged": sum(1 for event in events if not event.acknowledged),
        }
