import argparse
import json
from pathlib import Path

from surveilfusion.api.app import main as serve_main
from surveilfusion.core.config import get_settings
from surveilfusion.detectors.factory import build_detectors
from surveilfusion.detectors.service import DetectionService
from surveilfusion.onboarding import export_integration_configs, initialize_project, run_doctor
from surveilfusion.storage.events import EventStore
from surveilfusion.storage.notifications import NotificationStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="surveilfusion")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create local .env and camera config files.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing local config files.")

    doctor_parser = subparsers.add_parser("doctor", help="Check local setup readiness.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    export_parser = subparsers.add_parser(
        "export-integrations",
        help="Export go2rtc, Frigate, and Home Assistant config.",
    )
    export_parser.add_argument("--cameras-file", default="config/cameras.yml")
    export_parser.add_argument("--output-dir", default="generated")

    detect_parser = subparsers.add_parser("detect-image", help="Run configured detectors on an image file.")
    detect_parser.add_argument("image_path")
    detect_parser.add_argument("--camera-id", default="manual")
    detect_parser.add_argument("--json", action="store_true")

    notifications_parser = subparsers.add_parser("notifications", help="List local notification outbox entries.")
    notifications_parser.add_argument("--limit", type=int, default=20)
    notifications_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("serve", help="Run the SurveilFusion API and dashboard.")

    args = parser.parse_args()
    root = Path.cwd()

    if args.command == "init":
        created = initialize_project(root, force=args.force)
        if created:
            print("Created:")
            for path in created:
                print(f"  - {path}")
        else:
            print("Local config already exists. Use --force to overwrite.")
        return 0

    if args.command == "doctor":
        report = run_doctor(root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("SurveilFusion doctor")
            for name, ok in report["checks"].items():
                print(f"  {'OK' if ok else 'NO'}  {name}")
            print("Next steps:")
            for step in report["next_steps"]:
                print(f"  - {step}")
        return 0 if report["ok"] else 1

    if args.command == "export-integrations":
        cameras_file = Path(args.cameras_file)
        if not cameras_file.exists() and args.cameras_file == "config/cameras.yml":
            cameras_file = Path("config/cameras.example.yml")
        paths = export_integration_configs(cameras_file, Path(args.output_dir))
        print("Exported:")
        for path in paths:
            print(f"  - {path}")
        return 0

    if args.command == "detect-image":
        settings = get_settings()
        event_store = EventStore(settings.data_dir / "surveilfusion.db")
        service = DetectionService(build_detectors(settings), event_store)
        run = service.run_on_image_path_sync(Path(args.image_path), camera_id=args.camera_id)
        if args.json:
            print(run.model_dump_json(indent=2))
        else:
            print(f"{run.status.value}: {run.message}")
            for event in run.events:
                print(f"  - {event.title} ({event.severity.value})")
        return 0 if run.status.value != "failed" else 1

    if args.command == "notifications":
        settings = get_settings()
        store = NotificationStore(settings.data_dir / "surveilfusion.db")
        notifications = store.latest(limit=args.limit)
        if args.json:
            print(json.dumps([notification.model_dump(mode="json") for notification in notifications], indent=2))
        else:
            print("Notification outbox")
            for notification in notifications:
                print(f"  - {notification.status.value} {notification.channel.value}: {notification.title}")
        return 0

    if args.command in {None, "serve"}:
        serve_main()
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
