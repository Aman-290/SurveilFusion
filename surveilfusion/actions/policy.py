from surveilfusion.core.models import ActionCreate, ActionKind, ActionPolicyDecision, ActionRisk

HIGH_RISK_ACTIONS = {
    ActionKind.trigger_home_assistant_scene,
    ActionKind.move_ptz_preset,
    ActionKind.open_two_way_audio,
}

MEDIUM_RISK_ACTIONS = {
    ActionKind.start_recording,
    ActionKind.publish_mqtt,
}


class ActionPolicy:
    def evaluate(self, action: ActionCreate) -> ActionPolicyDecision:
        if action.kind in HIGH_RISK_ACTIONS:
            return ActionPolicyDecision(
                allowed=True,
                risk=ActionRisk.high,
                requires_approval=True,
                reason="High-impact remote control actions require human approval.",
            )
        if action.kind in MEDIUM_RISK_ACTIONS:
            return ActionPolicyDecision(
                allowed=True,
                risk=ActionRisk.medium,
                requires_approval=True,
                reason="Automation actions are queued for approval before execution.",
            )
        return ActionPolicyDecision(
            allowed=True,
            risk=ActionRisk.low,
            requires_approval=False,
            reason="Low-risk local action can execute without manual approval.",
        )
