# Security Policy

SurveilFusion handles sensitive home and workplace camera data. Treat every deployment as privacy-critical.

## Defaults

- Do not commit real camera URLs, passwords, bot tokens, face images, audio captures, or incident clips.
- Keep cloud LLM and remote tunnel integrations opt-in.
- Prefer LAN-only access, VPN, or zero-trust tunnels over public unauthenticated exposure.
- Set `SURVEILFUSION_API_KEY` before exposing the app outside localhost or your trusted LAN.
- Rotate credentials after testing demos.

## API Access

When `SURVEILFUSION_API_KEY` is configured, SurveilFusion protects `/api/*` and `/ws/*` with either:

```text
X-SurveilFusion-Key: your-key
```

or:

```text
Authorization: Bearer your-key
```

The dashboard shell and static assets remain available so local demos still load, but API calls fail until the key is supplied.

## Reporting

Open a private security advisory or contact the maintainer before publishing exploit details.
