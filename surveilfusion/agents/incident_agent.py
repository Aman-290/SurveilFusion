from surveilfusion.core.models import ActionCreate, ActionKind, AgentRecommendation, EventSeverity, SurveillanceEvent


class IncidentAgent:
    """Conservative triage agent with deterministic fallback behavior."""

    def recommend(self, event: SurveillanceEvent) -> AgentRecommendation:
        actions = {
            EventSeverity.critical: [
                "Notify all configured emergency contacts immediately.",
                "Pin the live camera feed and preserve the incident clip.",
                "Escalate to manual review if the alert is not acknowledged within 60 seconds.",
            ],
            EventSeverity.high: [
                "Notify configured contacts with snapshot and live feed.",
                "Record a 30 second before/after clip.",
                "Check nearby cameras for corroborating evidence.",
            ],
            EventSeverity.medium: [
                "Send a review notification to household admins.",
                "Compare against known-person memory before escalating.",
            ],
            EventSeverity.low: ["Log event and keep it searchable."],
            EventSeverity.info: ["Store for timeline context."],
        }
        rationale = f"{event.kind.value} event classified as {event.severity.value} from camera {event.camera_id}."
        return AgentRecommendation(
            event_id=event.id,
            priority=event.severity,
            rationale=rationale,
            recommended_actions=actions[event.severity],
            automation_plan=[
                "Create timeline entry",
                "Publish MQTT state update",
                "Queue notification if enabled",
            ],
        )

    def propose_actions(self, event: SurveillanceEvent) -> list[ActionCreate]:
        actions = [
            ActionCreate(
                kind=ActionKind.notify,
                camera_id=event.camera_id,
                event_id=event.id,
                reason=f"Notify configured contacts about {event.kind.value}.",
            ),
            ActionCreate(
                kind=ActionKind.pin_live_view,
                camera_id=event.camera_id,
                event_id=event.id,
                reason="Pin the relevant live camera while the incident is open.",
            ),
        ]
        if event.severity in {EventSeverity.high, EventSeverity.critical}:
            actions.append(
                ActionCreate(
                    kind=ActionKind.start_recording,
                    camera_id=event.camera_id,
                    event_id=event.id,
                    reason="Preserve a review clip for this high-priority incident.",
                )
            )
        return actions
