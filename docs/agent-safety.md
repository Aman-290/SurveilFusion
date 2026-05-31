# Agent And Remote Control Safety

SurveilFusion should support remote control and agentic automation without turning every alert into an unsafe action.

## Policy

- Agent recommendations are advisory by default.
- Remote actions must be represented as explicit, auditable action requests.
- High-impact actions require human acknowledgement before execution.
- Every action should include camera id, reason, requested by, created time, and resulting status.
- Cloud LLMs cannot receive raw camera frames unless explicitly enabled by the deployment owner.

## Early Action Types

- Notify household admin.
- Pin live view.
- Start recording window.
- Publish MQTT state.
- Trigger Home Assistant scene.
- Move PTZ preset.
- Open a two-way audio session.

## Guardrails

- Deny public unauthenticated remote control.
- Rate limit repeated actions.
- Log every action and policy decision.
- Prefer local deterministic rules for emergency escalation.
- Use LLMs for summarization and investigation plans, not direct emergency authority.
