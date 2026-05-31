# Contributing to SurveilFusion

SurveilFusion is aiming to be a practical local-first AI surveillance platform, not a cloud-only demo.

## Good first contribution areas

- Camera adapters for ONVIF, RTSP variants, USB cameras, and vendor quirks.
- Detector plugins for fire, smoke, person, vehicle, package, fall, and distress audio.
- Home Assistant, MQTT, Telegram, Matrix, email, and webhook integrations.
- Edge benchmarks for CPU, NVIDIA Jetson, Raspberry Pi AI Kit, Coral, and low-power mini PCs.
- Privacy, redaction, encryption, retention, and audit-log improvements.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Keep new features local-first by default and make cloud integrations opt-in.
