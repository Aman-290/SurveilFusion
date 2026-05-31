# Face Identity And Audio Detection

This rebuild intentionally avoids committing personal face images, raw audio captures, or model binaries. The new implementation adds clean local foundations that can be upgraded with stronger models.

## Face Identity

Implemented now:

- Local face identity enrollment by embedding.
- Local cosine-similarity matching.
- Unknown-face event generation.
- SQLite identity store.
- API endpoints:
  - `GET /api/identity/people`
  - `POST /api/identity/enroll`
  - `POST /api/identity/identify`

The embedding boundary is deliberate. It lets SurveilFusion support multiple face engines without rewriting the identity system.

Best upgrade candidates:

- InsightFace for high-quality detection, alignment, and ArcFace-style embeddings.
- DeepFace for a simple Python API across multiple face models.
- Frigate-style face recognition flow: run recognition on the detect stream, enroll only a few clean images per person, and keep it local.

Important caveat: face recognition is privacy-sensitive. Keep enrollment explicit, local, auditable, and easy to delete.

## Audio Detection

Implemented now:

- WAV analysis with duration, RMS dBFS, peak, and zero-crossing rate.
- Loud audio anomaly detection.
- Distress-audio event generation.
- API endpoint:
  - `POST /api/audio/analyze?audio_path=...&camera_id=...`
- CLI command:
  - `surveilfusion analyze-audio path\to\audio.wav --camera-id front-door --json`

Best upgrade candidates:

- Frigate-style CPU audio detection that only runs classifiers after volume crosses a threshold.
- PANNs / Efficient PANNs for AudioSet-style event classification.
- BEATs / OpenBEATs or CLAP-style encoders for richer audio understanding.
- Whisper-style transcription only for speech-heavy clips and only when explicitly enabled.

The current baseline is intentionally conservative. It detects loud anomalies locally and creates events, while leaving a clean path for real sound-event models.
