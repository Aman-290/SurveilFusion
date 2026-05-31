# SurveilFusion

Local-first AI surveillance for existing CCTV, RTSP, ONVIF, USB, and smart-home camera setups.

SurveilFusion turns ordinary cameras into a private AI security system: live local monitoring, fire/smoke and person detection, incident memory, agentic triage, MQTT/Home Assistant-ready automation, and optional local or cloud LLM assistance.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)
![Local First](https://img.shields.io/badge/Privacy-Local%20First-35d39a)

## Why This Exists

Most CCTV systems either record passively or push sensitive video into a vendor cloud. SurveilFusion is designed for the middle path: keep cameras and AI at home, add modern computer vision and agents, and make setup easy enough that people can clone the repo and get value quickly.

The original project proved the core idea with Flask, YOLO, YAMNet, face recognition, Telegram, WhatsApp, and Cloudflare Tunnel. This branch starts the 2026+ rebuild: typed services, Docker, cleaner APIs, memory, agent hooks, smart-home integration, and a path toward real contributor-friendly architecture.

## 2026+ Direction

- **Existing CCTV integration:** RTSP first, ONVIF discovery next, go2rtc/WebRTC for efficient browser live view.
- **Local hosting:** Docker Compose stack with app, MQTT, Qdrant, and Ollama.
- **Edge AI:** pluggable detector workers for YOLO-family models, ONNX/OpenVINO/TensorRT profiles, and low-FPS inference for cheap hardware.
- **Agentic incident response:** deterministic fallback now, optional local Ollama or OpenAI agent later for triage, summaries, and action plans.
- **Memory:** SQLite event timeline now, Qdrant vector memory planned for semantic search across incidents, clips, audio, and user preferences.
- **Automation:** MQTT payloads for Home Assistant and other local automations.
- **Privacy:** cloud services are opt-in, not required.

## Quick Start

```bash
git clone https://github.com/Aman-290/SurveilFusion.git
cd SurveilFusion
copy .env.example .env
docker compose up --build
```

Open `http://localhost:8080`.

For local Python development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m surveilfusion
```

## Configuration

Edit `config/cameras.example.yml` or set `CAMERAS_FILE` in `.env`.

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

Never commit real camera URLs, passwords, bot tokens, face images, or incident clips.

## API

- `GET /health` - service health
- `GET /api/cameras` - configured cameras
- `GET /api/events` - latest events
- `POST /api/events/demo` - create a synthetic event for setup testing
- `GET /api/events/{id}/recommendation` - incident agent recommendation
- `POST /api/events/{id}/ack` - acknowledge an event
- `GET /api/memory/summary` - local incident memory summary

FastAPI also exposes OpenAPI docs at `http://localhost:8080/docs`.

## Architecture

```mermaid
flowchart LR
  Cameras["RTSP / ONVIF / USB Cameras"] --> Gateway["go2rtc / stream gateway"]
  Gateway --> Workers["Detector workers"]
  Workers --> Events["Event store"]
  Events --> Agent["Incident agent"]
  Events --> Memory["SQLite + Qdrant memory"]
  Agent --> Notify["MQTT / Telegram / Webhooks"]
  Events --> Dashboard["FastAPI dashboard"]
```

## Research Notes

The modernization plan is based on current local AI and video infrastructure patterns:

- ONVIF Profile T covers advanced streaming, metadata, PTZ, motion regions, relay outputs, and bidirectional audio for compatible devices: [ONVIF Profile T](https://www.onvif.org/profiles/profile-t/).
- Frigate demonstrates the right local NVR posture: local object detection, real-time multiprocessing, and restreaming to reduce camera load: [Frigate docs](https://docs.frigate.video/).
- go2rtc/WebRTC is the practical live-view layer for RTSP camera setups: [Frigate go2rtc guide](https://docs.frigate.video/guides/configuring_go2rtc).
- Ultralytics YOLO11 improved efficiency and accuracy over YOLOv8, while current Ultralytics roadmap/platform material points toward newer YOLO26 edge-first models: [YOLO11 docs](https://docs.ultralytics.com/models/yolo11/).
- Modern agent systems increasingly include durable memory, tools, file/sandbox boundaries, and long-running workflows: [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/).
- Qdrant is a strong fit for local/edge incident memory and semantic retrieval: [Qdrant docs](https://qdrant.tech/documentation/).
- Ollama provides a local model runtime path for private agent features: [Ollama docs](https://docs.ollama.com/).

## Roadmap

See [docs/2026-modernization-plan.md](docs/2026-modernization-plan.md).

## Contributing

SurveilFusion needs camera adapters, detector plugins, Home Assistant examples, hardware benchmarks, and privacy/security hardening. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Camera systems are sensitive. Read [SECURITY.md](SECURITY.md) before exposing anything outside your LAN.

## License

MIT. Check model and third-party package licenses before commercial deployment.
