# Deployment Guide

SurveilFusion is designed to run locally first: mini PC, home server, NAS, Jetson, or a Raspberry Pi-class device.

## Docker Compose

```bash
surveilfusion init
docker compose up --build
```

Services:

- `surveilfusion`: FastAPI dashboard and API.
- `mqtt`: local Mosquitto broker for automation events.
- `qdrant`: vector database for future semantic incident memory.
- `ollama`: local LLM runtime for optional private agent features.

## Local Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
surveilfusion init
surveilfusion doctor
surveilfusion serve
```

## Hardware Targets

| Target | Recommended Use |
| --- | --- |
| Intel N100 mini PC | Best starter local server for multiple 1080p cameras at low detection FPS. |
| NVIDIA Jetson Orin Nano | Edge AI builds with TensorRT acceleration. |
| Raspberry Pi 5 + AI Kit | Lightweight local deployments and experimentation. |
| Desktop GPU server | Model experimentation, retraining, and high camera counts. |

## Production Notes

- Use a reverse proxy only after authentication is added.
- Keep `ENABLE_NOTIFICATIONS=false` until credentials are configured.
- Mount `data/`, `config/`, and `models/` as persistent volumes.
- Store model weights outside Git and fetch them through scripts or releases.
- Keep camera sources in local config files, never in committed examples.
