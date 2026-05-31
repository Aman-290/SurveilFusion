from surveilfusion.core.models import CameraConfig


def camera_discovery_payload(camera: CameraConfig, base_topic: str = "surveilfusion") -> dict[str, object]:
    unique_id = f"surveilfusion_{camera.id}_activity"
    state_topic = f"{base_topic}/cameras/{camera.id}/state"
    return {
        "name": f"{camera.name} Activity",
        "unique_id": unique_id,
        "state_topic": state_topic,
        "payload_on": "active",
        "payload_off": "idle",
        "device_class": "motion",
        "device": {
            "identifiers": [f"surveilfusion_{camera.id}"],
            "name": camera.name,
            "manufacturer": "SurveilFusion",
            "model": "Local AI Camera",
        },
    }


def discovery_messages(cameras: list[CameraConfig], base_topic: str = "surveilfusion") -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for camera in cameras:
        messages.append(
            {
                "topic": f"homeassistant/binary_sensor/surveilfusion/{camera.id}/config",
                "payload": camera_discovery_payload(camera, base_topic=base_topic),
                "retain": True,
            }
        )
    return messages
