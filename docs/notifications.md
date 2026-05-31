# Notifications

SurveilFusion uses an auditable notification outbox before sending anything externally.

## Implemented Now

- Event notifications are represented as `NotificationMessage` records.
- Demo events queue local outbox notifications automatically.
- Notifications are persisted in SQLite.
- Dashboard shows the outbox.
- Dispatch boundary supports local outbox and webhook channels.

## API

- `GET /api/notifications`
- `POST /api/events/{event_id}/notifications/queue`
- `POST /api/notifications/{notification_id}/dispatch`

## CLI

```bash
surveilfusion notifications --json
```

## Dispatch Philosophy

The outbox is intentionally first-class. Security systems should not silently send or fail to send alerts without a record. Each future channel should preserve:

- target
- payload
- created time
- sent time
- status
- error details

## Next Channels

- Telegram bot delivery.
- MQTT/Home Assistant alert topics.
- Email and generic webhooks.
- Optional WhatsApp provider through a documented adapter.
