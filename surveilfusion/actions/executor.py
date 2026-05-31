from datetime import datetime, timezone

from surveilfusion.core.models import ActionKind, ActionRequest, ActionStatus


class ActionExecutor:
    """Deterministic local executor. Real integrations plug in behind this boundary."""

    def execute(self, action: ActionRequest) -> ActionRequest:
        if action.requires_approval and action.status != ActionStatus.approved:
            action.status = ActionStatus.failed
            action.result = "Action requires approval before execution."
            return action

        action.status = ActionStatus.executed
        action.executed_at = datetime.now(timezone.utc)
        action.result = EXECUTION_RESULTS[action.kind]
        return action


EXECUTION_RESULTS = {
    ActionKind.notify: "Notification queued for configured channels.",
    ActionKind.pin_live_view: "Live view pinned in the command center.",
    ActionKind.start_recording: "Recording window requested for this camera.",
    ActionKind.publish_mqtt: "MQTT action payload queued.",
    ActionKind.trigger_home_assistant_scene: "Home Assistant scene trigger queued.",
    ActionKind.move_ptz_preset: "PTZ preset move queued.",
    ActionKind.open_two_way_audio: "Two-way audio session requested.",
}
