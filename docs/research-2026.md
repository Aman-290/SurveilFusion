# 2026 Research Notes

The next useful SurveilFusion upgrades should follow proven local-first patterns rather than adding random AI features.

## Camera And NVR Layer

- Frigate remains a strong reference for local object detection, audio events, face recognition, and Home Assistant integration: https://docs.frigate.video/
- Scrypted is useful inspiration for camera compatibility and bridging streams to HomeKit, Google Home, Alexa, and Home Assistant: https://docs.scrypted.app/
- go2rtc is the practical stream fan-out layer for RTSP/WebRTC so multiple clients do not overload cameras: https://github.com/AlexxIT/go2rtc

## Face Identity

- InsightFace is a strong open-source face analysis toolkit for detection, alignment, and recognition, but model/data licensing must be checked carefully before commercial use: https://github.com/deepinsight/insightface
- DeepFace is easier to integrate and wraps multiple face recognition backends: https://github.com/serengil/deepface
- Frigate's face recognition docs reinforce the product lesson: train from a few clean images per person and keep recognition close to the detect stream: https://docs.frigate.video/configuration/face_recognition/
- The best architecture is to keep identity storage separate from model choice. Store local embeddings and metadata; let the detector backend be swappable.

## Audio Events

- Frigate's audio detector pattern is smart: use volume as a gate before running classification to save CPU: https://docs.frigate.video/configuration/audio_detectors/
- PANNs and Efficient PANNs are practical AudioSet-style classification baselines: https://github.com/qiuqiangkong/audioset_tagging_cnn
- BEATs/OpenBEATs and CLAP-style encoders are stronger future paths for richer audio understanding: https://github.com/microsoft/unilm/tree/master/beats

## What We Should Avoid

- Do not store personal faces or audio in the repo.
- Do not make cloud AI mandatory.
- Do not add face recognition without enrollment, deletion, audit logs, and privacy notes.
- Do not pretend a simple loudness detector is a full scream classifier.

## Product Direction

SurveilFusion should be a clean local AI security platform:

1. Easy camera onboarding.
2. Local detector workers.
3. Searchable event memory.
4. Policy-gated remote actions.
5. Auditable notifications.
6. Optional pluggable AI backends for face, audio, and LLM summarization.
