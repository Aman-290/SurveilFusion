# Camera Onboarding

SurveilFusion should make existing cameras useful without replacing hardware. Start with RTSP because it works across many CCTV brands, then add ONVIF discovery and vendor-specific presets as the project matures.

## Fast Path

1. Run `surveilfusion init`.
2. Edit `config/cameras.yml`.
3. Run `surveilfusion export-integrations`.
4. Start the app with `docker compose up --build`.
5. Open `http://localhost:8080`.

## RTSP URL Examples

```yaml
cameras:
  - id: front-door
    name: Front Door
    source: rtsp://user:password@192.168.1.20:554/stream1
    zone: entrance
    detect_fps: 5
    record: true
    audio: true
```

## Vendor Notes

- Reolink commonly exposes `h264Preview_01_main` or `h264Preview_01_sub` style RTSP paths.
- Hikvision commonly exposes `Streaming/Channels/101` for main stream and `Streaming/Channels/102` for substream.
- Dahua commonly exposes `cam/realmonitor?channel=1&subtype=0`.
- Tapo, Wyze Bridge, and many NVRs expose RTSP after enabling it in device settings.

Always use a low-resolution substream for detection if available. Keep the main stream for recording or high-quality snapshots.

## Generated Integrations

`surveilfusion export-integrations` writes:

- `generated/go2rtc.yml` for RTSP restreaming and future WebRTC live view.
- `generated/frigate.cameras.yml` for users who want to compare or run alongside Frigate.
- `generated/home-assistant-mqtt-discovery.json` for Home Assistant MQTT entities.

## Privacy Checklist

- Put cameras and SurveilFusion on the same LAN or VLAN.
- Use unique camera passwords.
- Do not expose RTSP directly to the internet.
- Prefer VPN or zero-trust tunnel access for remote viewing.
- Keep cloud LLMs disabled unless the deployment owner explicitly opts in.
