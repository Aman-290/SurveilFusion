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

## Implemented Action Flow

SurveilFusion now represents remote control as an auditable action request:

1. Agent or operator creates an action.
2. Policy assigns risk and approval requirements.
3. Low-risk actions can execute immediately.
4. Medium and high-risk actions must be approved before execution.
5. Every status change is stored in the local SQLite database.

Current API surface:

- `GET /api/actions`
- `POST /api/actions`
- `POST /api/actions/{id}/approve`
- `POST /api/actions/{id}/deny`
- `POST /api/actions/{id}/execute`
- `POST /api/events/{id}/actions/propose`

## Guardrails

- Deny public unauthenticated remote control.
- Set `SURVEILFUSION_API_KEY` for any deployment where remote actions are reachable beyond a trusted LAN.
- Rate limit repeated actions.
- Log every action and policy decision.
- Prefer local deterministic rules for emergency escalation.
- Use LLMs for summarization and investigation plans, not direct emergency authority.
