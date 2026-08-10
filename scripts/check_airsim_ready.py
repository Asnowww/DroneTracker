"""Preflight check before any AirSim run. Exits non-zero if the rig is not ready.

    python scripts/check_airsim_ready.py --config config/tracking_config.json \
        --weights runs/detect/drone_airsim/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import load_json  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "tracking_config.json"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--dataset-config", default=str(ROOT / "config" / "dataset_config.json"))
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    tracker, target = cfg["tracker_vehicle"], cfg["target_vehicle"]
    camera = str(cfg["camera_name"])
    width, height = int(cfg["image_width"]), int(cfg["image_height"])

    # --- python side -------------------------------------------------------
    try:
        import torch

        cuda = torch.cuda.is_available()
        record(
            "torch CUDA",
            cuda,
            torch.cuda.get_device_name(0) if cuda else "CPU only — inference will be far slower",
        )
    except ImportError as exc:
        record("torch", False, str(exc))

    try:
        import ultralytics

        record("ultralytics", True, ultralytics.__version__)
    except ImportError as exc:
        record("ultralytics", False, str(exc))

    if args.weights:
        weights_path = Path(args.weights)
        record("weights file", weights_path.exists(), str(weights_path))

    try:
        from drone_tracker.airsim_io import (
            configure_segmentation,
            connect,
            get_scene_and_depth,
            list_scene_objects,
            list_vehicles,
            module_name,
            require_vehicles,
        )

        record("airsim client import", True, module_name())
    except ImportError as exc:
        record("airsim client import", False, str(exc))
        report_and_exit()
        return

    # --- simulator side ----------------------------------------------------
    try:
        client = connect(timeout_s=15.0)
        record("RPC connect", True, "127.0.0.1:41451")
    except Exception as exc:  # noqa: BLE001
        record("RPC connect", False, str(exc))
        report_and_exit()
        return

    vehicles = list_vehicles(client)
    try:
        require_vehicles(client, [tracker, target])
        record("vehicles", True, f"{vehicles or [tracker, target]}")
    except Exception as exc:  # noqa: BLE001
        record("vehicles", False, str(exc))

    try:
        client.simSetCameraFov(camera, float(cfg["camera_fov_deg"]), vehicle_name=tracker)
        record("camera FOV set", True, f"{cfg['camera_fov_deg']} deg on camera {camera!r}")
    except Exception as exc:  # noqa: BLE001
        record("camera FOV set", False, str(exc))

    frame = get_scene_and_depth(client, camera, tracker)
    if frame is None:
        record("scene capture", False, "simGetImages returned nothing")
    else:
        h, w = frame.rgb.shape[:2]
        record(
            "scene capture",
            (w, h) == (width, height),
            f"got {w}x{h}, config expects {width}x{height}"
            + ("" if (w, h) == (width, height) else "  <-- fix settings.json CaptureSettings"),
        )
        record(
            "depth capture",
            frame.depth is not None,
            "DepthPlanar present" if frame.depth is not None else "no depth — add ImageType 2 to settings.json",
        )

    if not args.skip_segmentation and Path(args.dataset_config).exists():
        from drone_tracker.airsim_io import get_target_instance_colors

        seg_cfg = load_json(args.dataset_config)["segmentation"]
        mesh_regex = seg_cfg.get("target_mesh_regex", seg_cfg.get("target_regex", "Target"))
        colors = get_target_instance_colors(client, mesh_regex)
        if colors is not None:
            record(
                "segmentation target",
                bool(colors),
                f"instance scheme: mesh_regex {mesh_regex!r} -> {len(colors)} meshes"
                + ("" if colors else "  <-- run scripts/list_scene_objects.py --instance"),
            )
        else:
            regex = seg_cfg["target_regex"]
            matched = list_scene_objects(client, regex)
            ok = configure_segmentation(client, regex, int(seg_cfg["target_id"]))
            record(
                "segmentation target",
                bool(ok and matched),
                f"painted-ID scheme: regex {regex!r} matched {matched or '[]'}"
                + ("" if matched else "  <-- run scripts/list_scene_objects.py"),
            )

    report_and_exit()


def report_and_exit() -> None:
    width = max((len(name) for name, _, _ in CHECKS), default=10) + 2
    print()
    for name, ok, detail in CHECKS:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<{width}} {detail}")
    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failed:
        print(f"NOT READY — failing checks: {', '.join(failed)}")
        sys.exit(1)
    print("airsim_ready=true")
    sys.exit(0)


if __name__ == "__main__":
    main()
