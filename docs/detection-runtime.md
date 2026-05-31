# Detection Runtime

SurveilFusion now has a detector orchestration layer instead of only standalone model wrappers.

## What Works Now

- `DetectionService` runs one or more detector plugins on a decoded frame.
- Detections are converted into `SurveillanceEvent` records.
- Events are persisted in the local SQLite event store.
- API and CLI entry points can run configured detectors on an image file.
- The service is testable without downloading large model weights.

## CLI

```bash
surveilfusion detect-image path\to\snapshot.jpg --camera-id front-door --json
```

If `models/fire-yolo.pt` exists, SurveilFusion wires the YOLO detector automatically. Otherwise the command reports that no detectors are configured or that the vision extra/model is missing.

## API

- `GET /api/detect/status`
- `POST /api/detect/image?image_path=path/to/snapshot.jpg&camera_id=front-door`

## Model Path

Set `FIRE_MODEL_PATH` in `.env` or place a detector at:

```text
models/fire-yolo.pt
```

Model files are intentionally excluded from Git. Publish trained weights through releases or a documented download script, not as repository blobs.

## Next Runtime Steps

- RTSP frame sampling worker per camera.
- Zone masks and per-camera detector settings.
- Object tracking and cooldowns to avoid duplicate alerts.
- ONNX, OpenVINO, TensorRT, and Hailo export profiles.
- Snapshot and clip attachment for created events.
