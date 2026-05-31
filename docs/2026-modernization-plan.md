# SurveilFusion 2026+ Modernization Plan

## Product Thesis

SurveilFusion should become a local-first AI CCTV command center that upgrades existing home and small-business cameras without forcing users into another cloud camera subscription. The viral wedge is simple: clone, run Docker, add your RTSP/ONVIF cameras, get private AI alerts, searchable incident memory, Home Assistant/MQTT automation, and optional agentic investigation.

## What Changed By 2026

- Edge vision is now expected to be efficient and model-flexible. Ultralytics positions YOLO11 as more accurate and efficient than YOLOv8, and its current roadmap/platform messaging points toward YOLO26 for edge-first deployment.
- Local NVR users expect Frigate-style architecture: multiprocessing, RTSP restreaming, Home Assistant integration, and local object detection instead of cloud dependence.
- Camera integration should start with RTSP and ONVIF Profile T/M concepts: video streaming, metadata/events, PTZ, motion regions, relay outputs, and bidirectional audio where supported.
- Browser viewing should use go2rtc/WebRTC or MSE for low-latency live view and to avoid multiple direct camera connections.
- AI agents are moving from one-shot prompts to tool-using systems with durable memory, sandboxing, file outputs, and long-running workflows.
- Local LLMs through Ollama and vector memory through Qdrant are mainstream enough to be first-class optional services.

## Architecture

1. FastAPI control plane with typed Pydantic settings and OpenAPI docs.
2. Camera registry backed by YAML first, then UI onboarding and ONVIF discovery.
3. Stream gateway based on go2rtc for RTSP/WebRTC fan-out.
4. Detector workers for fire/smoke, person/vehicle/package, face memory, and distress audio.
5. Event store with snapshots, clips, acknowledgement state, retention, and audit logs.
6. Agent layer for incident triage, correlation across cameras, notification drafting, and action plans.
7. Memory layer with SQLite for event history and Qdrant for semantic search over incidents, clips, transcripts, and user preferences.
8. Integrations: MQTT/Home Assistant discovery, Telegram, webhooks, email, optional WhatsApp provider.
9. Deployment: Docker Compose for local hosting, optional GPU profile, documented mini-PC/Jetson/Raspberry Pi targets.

## Implementation Roadmap

### Phase 1: Foundation

- Replace the monolithic Flask script with a typed FastAPI app.
- Add Dockerfile, Docker Compose, `.env.example`, CI, tests, and clean docs.
- Create API endpoints for cameras, events, memory summaries, acknowledgements, and agent recommendations.
- Keep old prototype code available as historical reference until feature parity is reached.

### Phase 2: Real Camera Onboarding

- Add RTSP validation and snapshot capture.
- Add ONVIF discovery and credential testing.
- Generate go2rtc config from the camera registry.
- Add WebRTC live tiles in the dashboard.

### Phase 3: Detector Runtime

- Run detector workers outside the web process.
- Support model profiles: CPU, CUDA, OpenVINO, TensorRT, CoreML, and ONNX Runtime.
- Add cooldowns, zone masks, object tracking, and confidence calibration.
- Remove checked-in model binaries and fetch models via scripts.

### Phase 4: Agents And Memory

- Add incident agent with pluggable providers: disabled, Ollama, OpenAI.
- Store embeddings in Qdrant for semantic incident search.
- Add camera-aware memory: recurring false positives, usual occupants, risk windows, and location context.
- Add agent guardrails so remote actions require policy approval.

### Phase 5: Community And Growth

- Improve README SEO for AI surveillance, local CCTV AI, RTSP, ONVIF, Home Assistant, Docker, and edge AI.
- Add screenshots, demo video, architecture diagram, hardware benchmark table, and comparison with Frigate/Scrypted/Home Assistant setups.
- Add GitHub issue templates, labels, contribution guide, security policy, and project board.
- Publish example configs for Reolink, Hikvision, Dahua, Tapo, Wyze bridge, and generic RTSP.

## Immediate Code Delivered In This Branch

- New FastAPI application package.
- Local event store and incident memory summary.
- Deterministic incident agent fallback.
- MQTT payload conversion.
- Docker Compose stack with SurveilFusion, MQTT, Qdrant, and Ollama.
- Responsive dashboard and demo event flow.
- CI, tests, contribution guide, and security policy.
