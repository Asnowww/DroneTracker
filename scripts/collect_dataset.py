"""Collect a YOLO drone-detection dataset from AirSim with zero manual labelling.

Every frame is labelled from the *same* segmentation capture that produced the RGB
image, so there is no pose/image synchronisation error. A ground-truth cuboid
projection runs alongside it as a cross-check and rejects frames where the two
disagree.

    python scripts/collect_dataset.py --config config/dataset_config.json --samples 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.airsim_io import (  # noqa: E402
    airsim_module,
    configure_segmentation,
    quaternion_from_euler,
    teleport_still,
    connect,
    get_camera_info,
    get_scene_and_segmentation,
    get_scene_depth_and_segmentation,
    get_target_instance_colors,
    hover_lock,
    line_of_sight_clear,
    list_scene_objects,
    pose_xyz_yaw,
    require_vehicles,
    set_camera_gimbal,
    set_time_of_day,
    set_weather,
)
from drone_tracker.config import load_json, save_json  # noqa: E402
from drone_tracker.labeling import (  # noqa: E402
    bbox_and_pixels_from_segmentation,
    bbox_from_projection,
    bbox_from_segmentation,
    bbox_iou,
    degenerate_frame_reason,
    label_box_looks_empty,
    quaternion_to_matrix,
    to_yolo_line,
    touches_border,
)
from drone_tracker.utils import focal_px  # noqa: E402


def pose_with_rpy(x: float, y: float, z: float, roll_deg: float, pitch_deg: float, yaw_deg: float):
    airsim = airsim_module()
    return airsim.Pose(
        airsim.Vector3r(float(x), float(y), float(z)),
        quaternion_from_euler(
            math.radians(float(pitch_deg)), math.radians(float(roll_deg)), math.radians(float(yaw_deg))
        ),
    )


def target_world_from_relative(
    tracker_x: float,
    tracker_y: float,
    tracker_z: float,
    tracker_yaw_deg: float,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float]:
    """Place the target at a spherical offset in the tracker's body frame (NED)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    psi = math.radians(tracker_yaw_deg)
    x_b = distance_m * math.cos(el) * math.cos(az)
    y_b = distance_m * math.cos(el) * math.sin(az)
    z_b = -distance_m * math.sin(el)
    return (
        tracker_x + x_b * math.cos(psi) - y_b * math.sin(psi),
        tracker_y + x_b * math.sin(psi) + y_b * math.cos(psi),
        tracker_z + z_b,
    )


def setup_segmentation(
    client, seg_cfg: dict, tracker: str, target: str, camera: str, fov_deg: float
) -> tuple[list[tuple[int, int, int]] | None, bool, dict | None]:
    """Pick a labelling scheme; return ``(target_colors, ok, calibration)``.

    Cosys-AirSim 3.4+: instance segmentation gives every mesh a fixed colour — look
    up the target meshes' colours, then calibrate against one known placement:
    which channel order the frames use, the mesh's *actual* visual extents, and the
    offset between the vehicle origin and the visual centre. The measured values
    drive the projection cross-check, so it agrees with the mask up to real
    disagreements (occlusion, wrong mesh) instead of failing on cuboid guesswork.

    Old AirSim: paint the scene id 0 / target id 25 and use border-modal masking
    (``target_colors`` stays None, calibration None).
    """
    mesh_regex = seg_cfg.get("target_mesh_regex", seg_cfg.get("target_regex", "Target"))
    colors = get_target_instance_colors(client, mesh_regex)

    if colors is None:  # old painted-ID API
        matched = list_scene_objects(client, seg_cfg["target_regex"])
        print(f"painted-ID scheme: regex={seg_cfg['target_regex']!r} matched={matched}")
        ok = configure_segmentation(client, seg_cfg["target_regex"], int(seg_cfg["target_id"]))
        return None, ok, None

    if not colors:
        print(f"instance segmentation: mesh_regex={mesh_regex!r} matched no meshes")
        return None, False, None
    print(f"instance segmentation: mesh_regex={mesh_regex!r} -> {len(colors)} meshes, colors={colors}")

    # Calibration: park the target dead ahead and measure it.
    teleport_still(client, tracker, 0.0, 0.0, -30.0, 0.0)
    teleport_still(client, target, 8.0, 0.0, -30.0, 180.0)
    set_camera_gimbal(client, tracker, camera, 0.0, 0.0)
    time.sleep(float(seg_cfg.get("settle_s", 0.12)) + 0.2)
    frame = get_scene_and_segmentation(client, camera, tracker)
    if frame is None or frame.segmentation is None:
        print("calibration capture failed")
        return None, False, None

    flipped = [(b, g, r) for r, g, b in colors]
    seg = frame.segmentation.astype(np.uint32)
    packed = (seg[:, :, 0] << 16) | (seg[:, :, 1] << 8) | seg[:, :, 2]

    def count(color_list: list[tuple[int, int, int]]) -> int:
        wanted = np.array([(r << 16) | (g << 8) | b for r, g, b in color_list], dtype=np.uint32)
        return int(np.isin(packed, wanted).sum())

    as_is, reversed_ = count(colors), count(flipped)
    print(f"calibration: pixels as-is={as_is} channel-reversed={reversed_}")
    if max(as_is, reversed_) < 16:
        print("calibration failed: target colours not found in the frame")
        return None, False, None
    final_colors = colors if as_is >= reversed_ else flipped

    # Measure the mesh from the calibration frame: apparent size + vertical offset
    # of the visual centre relative to the vehicle-origin projection.
    calibration: dict | None = None
    bbox, cal_pixels = bbox_and_pixels_from_segmentation(
        frame.segmentation, min_area_px=16, target_colors=final_colors
    )
    if bbox is not None:
        cam_info = get_camera_info(client, camera, tracker)
        cam_pos, cam_rot = camera_pose_arrays(cam_info)
        tp = client.simGetVehiclePose(target)
        tgt_pos = np.array([tp.position.x_val, tp.position.y_val, tp.position.z_val], dtype=float)
        height, width = frame.segmentation.shape[:2]
        from drone_tracker.labeling import project_point
        from drone_tracker.utils import focal_px

        origin_uv = project_point(tgt_pos, cam_pos, cam_rot, fov_deg, width, height)
        if origin_uv is not None:
            distance = float(np.linalg.norm(tgt_pos - cam_pos))
            f = focal_px(fov_deg, width)
            width_m = (bbox[2] - bbox[0]) * distance / f
            height_m = (bbox[3] - bbox[1]) * distance / f
            offset_z_m = ((bbox[1] + bbox[3]) / 2.0 - origin_uv[1]) * distance / f
            # Front view only: assume x/y symmetry, pad for unseen depth extent.
            calibration = {
                "extents_m": (width_m * 1.2, width_m * 1.2, max(height_m * 1.3, 0.05)),
                "offset_z_m": offset_z_m,
                # Visible-fraction reference: an unoccluded target at distance d is
                # expected to cover ~ref_pixels * (ref_distance/d)^2 mask pixels.
                "ref_pixels": int(cal_pixels),
                "ref_distance_m": distance,
            }
            print(
                f"calibration: measured width={width_m:.2f}m height={height_m:.2f}m "
                f"visual-centre offset_z={offset_z_m:+.2f}m -> extents={calibration['extents_m']}"
            )

    return final_colors, True, calibration


def camera_pose_arrays(camera_info):
    position = camera_info.pose.position
    orientation = camera_info.pose.orientation
    return (
        np.array([position.x_val, position.y_val, position.z_val], dtype=float),
        quaternion_to_matrix(
            orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val
        ),
    )


def randomize_environment(client, rnd: random.Random, cfg: dict) -> None:
    airsim = airsim_module()
    if rnd.random() < float(cfg.get("weather_prob", 0.0)):
        intensity = rnd.uniform(0.0, float(cfg.get("weather_max_intensity", 0.4)))
        choice = rnd.choice(["Rain", "Snow", "Fog", "Dust"])
        set_weather(client, True, **{choice: intensity})
    else:
        try:
            client.simEnableWeather(False)
        except Exception:  # noqa: BLE001
            pass
    if rnd.random() < float(cfg.get("time_of_day_prob", 0.0)):
        lo, hi = cfg.get("time_of_day_hours", [6, 20])
        hour = rnd.randint(int(lo), int(hi))
        minute = rnd.randint(0, 59)
        set_time_of_day(client, True, f"2024-06-21 {hour:02d}:{minute:02d}:00")
    else:
        set_time_of_day(client, False)
    del airsim


def write_preview(image, bbox, path: Path) -> None:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return
    canvas = image.copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "dataset_config.json"))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--preview", type=int, default=24, help="annotated sanity-check images to dump")
    parser.add_argument(
        "--top-up",
        action="store_true",
        help="append to an existing dataset: keep numbering after the highest "
        "existing frame index and collect --samples ADDITIONAL frames",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="freeze sim physics while capturing (old AirSim only: Cosys-AirSim 3.4 "
        "silently ignores simSetVehiclePose while paused, which would freeze every "
        "frame at the calibration scene)",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    seg_cfg = cfg["segmentation"]
    rnd_cfg = cfg["randomization"]
    samples_target = int(args.samples if args.samples is not None else cfg["samples"])
    out_root = Path(args.output or cfg["output_dir"])
    width, height = int(cfg["image_width"]), int(cfg["image_height"])
    fov_deg = float(cfg["camera_fov_deg"])
    camera = str(cfg["camera_name"])
    tracker, target = cfg["tracker_vehicle"], cfg["target_vehicle"]

    rnd = random.Random(int(cfg.get("seed", 0)))

    client = connect()
    require_vehicles(client, [tracker, target])
    client.simSetCameraFov(camera, fov_deg, vehicle_name=tracker)
    set_camera_gimbal(client, tracker, camera, 0.0, 0.0)
    # Keep both drones actively hovering so teleports don't turn into free fall.
    hover_lock(client, [tracker, target])
    time.sleep(1.0)

    target_colors, seg_ok, calibration = setup_segmentation(
        client, seg_cfg, tracker, target, camera, fov_deg
    )
    if not seg_ok:
        print(
            "WARNING: no usable segmentation scheme. Falling back to cuboid "
            "projection labels. Run scripts/list_scene_objects.py to find the real mesh name."
        )

    if args.pause:
        try:
            client.simPause(True)
            # Guard against servers that ignore teleports while paused (Cosys 3.4).
            probe = client.simGetVehiclePose(target)
            client.simSetVehiclePose(
                pose_xyz_yaw(probe.position.x_val + 1.0, probe.position.y_val, probe.position.z_val, 0.0),
                True,
                target,
            )
            time.sleep(0.05)
            moved = abs(client.simGetVehiclePose(target).position.x_val - probe.position.x_val) > 0.5
            if not moved:
                client.simPause(False)
                print("WARNING: this server ignores teleports while paused — continuing unpaused")
                args.pause = False
        except Exception:  # noqa: BLE001
            print("WARNING: simPause unsupported, physics will drift slightly between frames")

    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("opencv-python is required to write dataset images") from exc

    val_ratio = float(cfg["val_ratio"])
    background_ratio = float(cfg.get("background_ratio", 0.0))
    settle_s = float(seg_cfg.get("settle_s", 0.12))
    min_area = int(seg_cfg.get("min_box_area_px", 24))
    cross_check = bool(seg_cfg.get("cross_check_projection", True))
    min_iou = float(seg_cfg.get("min_cross_check_iou", 0.25))
    if calibration is not None:
        extents = calibration["extents_m"]
        target_offset_z_m = float(calibration["offset_z_m"])
    else:
        extents = tuple(float(v) for v in seg_cfg.get("target_extents_m", [1.0, 1.0, 0.35]))
        target_offset_z_m = 0.0

    # Visible-fraction gate: expected mask pixels at distance d, from calibration.
    ref_pixels = int(calibration["ref_pixels"]) if calibration else 0
    ref_distance = float(calibration["ref_distance_m"]) if calibration else 0.0
    min_visible_fraction = float(seg_cfg.get("min_visible_fraction", 0.3))

    stats = {
        "requested": samples_target,
        "written": 0,
        "background": 0,
        "rejected_no_mask": 0,
        "rejected_small": 0,
        "rejected_cross_check": 0,
        "rejected_capture": 0,
        "rejected_los": 0,
        "rejected_degenerate": 0,
        "rejected_occluded": 0,
        "rejected_empty_box": 0,
        "cross_check_iou_sum": 0.0,
        "cross_check_iou_n": 0,
        "label_source": "segmentation" if seg_ok else "projection",
    }
    previews_left = int(args.preview)
    attempts = 0
    max_attempts = samples_target * 6
    start_time = time.time()

    start_index = 0
    if args.top_up:
        existing = [
            int(m.group(1))
            for split in ("train", "val")
            for p in (images_dir / split).glob("frame_*.jpg")
            if (m := re.match(r"frame_(\d+)", p.stem))
        ]
        start_index = max(existing) + 1 if existing else 0
        print(f"top-up mode: {len(existing)} existing frames, numbering from {start_index}")

    while stats["written"] < samples_target and attempts < max_attempts:
        attempts += 1
        want_background = rnd.random() < background_ratio

        tracker_x = rnd.uniform(*rnd_cfg["tracker_x_m"])
        tracker_y = rnd.uniform(*rnd_cfg["tracker_y_m"])
        tracker_z = -rnd.uniform(*rnd_cfg["tracker_altitude_m"])
        tracker_yaw = rnd.uniform(*rnd_cfg["tracker_yaw_deg"])
        teleport_still(client, tracker, tracker_x, tracker_y, tracker_z, tracker_yaw)

        # Log-uniform: the visual-servo operating point is 3-8 m, so bias sampling
        # toward close range instead of letting 3-45 m be dominated by far frames.
        d_lo, d_hi = rnd_cfg["distance_m"]
        distance = math.exp(rnd.uniform(math.log(d_lo), math.log(d_hi)))
        if want_background:
            # Park the target far behind the camera so the frame is a true negative.
            target_x, target_y, target_z = target_world_from_relative(
                tracker_x, tracker_y, tracker_z, tracker_yaw + 180.0, 60.0, 0.0, 0.0
            )
        else:
            target_x, target_y, target_z = target_world_from_relative(
                tracker_x,
                tracker_y,
                tracker_z,
                tracker_yaw,
                distance,
                rnd.uniform(*rnd_cfg["azimuth_deg"]),
                rnd.uniform(*rnd_cfg["elevation_deg"]),
            )
        teleport_still(
            client,
            target,
            target_x,
            target_y,
            target_z,
            yaw_deg=rnd.uniform(*rnd_cfg["target_yaw_deg"]),
            pitch_deg=rnd.uniform(*rnd_cfg["target_pitch_deg"]),
            roll_deg=rnd.uniform(*rnd_cfg["target_roll_deg"]),
        )

        # Layer 1a: don't even capture if a building blocks the camera-target ray.
        if not want_background:
            los = line_of_sight_clear(
                client, (tracker_x, tracker_y, tracker_z), (target_x, target_y, target_z)
            )
            if los is False:
                stats["rejected_los"] += 1
                continue

        randomize_environment(client, rnd, rnd_cfg)
        time.sleep(settle_s)

        # Double capture: Scene and Segmentation are separate render passes that can
        # desynchronise after a teleport (the seg pass may still show the PREVIOUS
        # placement). The first capture flushes the stale pass; only the second is
        # trusted.
        get_scene_depth_and_segmentation(client, camera, tracker)
        frame = get_scene_depth_and_segmentation(client, camera, tracker)
        if frame is None or frame.rgb is None:
            stats["rejected_capture"] += 1
            continue

        # Layer 1b/2: flat-colour frames and camera-inside-geometry captures are
        # garbage for positives AND negatives alike.
        degenerate = degenerate_frame_reason(frame.rgb, frame.depth)
        if degenerate is not None:
            stats["rejected_degenerate"] += 1
            continue

        bbox = None
        if not want_background:
            if seg_ok:
                bbox, mask_pixels = bbox_and_pixels_from_segmentation(
                    frame.segmentation, min_area_px=min_area, target_colors=target_colors
                )
                if bbox is None:
                    stats["rejected_no_mask"] += 1
                    continue
                # Layer 2: partial occlusion — a sliver of drone peeking past an
                # edge has a valid mask but far fewer pixels than an unoccluded
                # target would have at this distance.
                if ref_pixels > 0 and distance > 0:
                    expected = ref_pixels * (ref_distance / distance) ** 2
                    if expected >= min_area and mask_pixels < min_visible_fraction * expected:
                        stats["rejected_occluded"] += 1
                        continue
            if cross_check or not seg_ok:
                camera_info = get_camera_info(client, camera, tracker)
                cam_pos, cam_rot = camera_pose_arrays(camera_info)
                target_pose = client.simGetVehiclePose(target)
                tgt_pos = np.array(
                    [
                        target_pose.position.x_val,
                        target_pose.position.y_val,
                        target_pose.position.z_val,
                    ],
                    dtype=float,
                )
                tgt_rot = quaternion_to_matrix(
                    target_pose.orientation.x_val,
                    target_pose.orientation.y_val,
                    target_pose.orientation.z_val,
                    target_pose.orientation.w_val,
                )
                # Shift the cuboid to the measured visual centre of the mesh.
                tgt_visual_pos = tgt_pos + tgt_rot @ np.array([0.0, 0.0, target_offset_z_m])
                projected = bbox_from_projection(
                    tgt_visual_pos, tgt_rot, extents, cam_pos, cam_rot, fov_deg, width, height
                )
                if bbox is None:
                    bbox = projected
                    if bbox is None:
                        stats["rejected_no_mask"] += 1
                        continue
                elif projected is not None:
                    iou = bbox_iou(bbox, projected)
                    if target_colors is not None:
                        # Instance-segmentation mask colours are unique per mesh, so
                        # the mask cannot belong to another object. The projection
                        # runs on the physics clock while the capture runs on the
                        # render clock, and Cosys 3.4 lets them drift apart — so the
                        # IoU is recorded as a health metric, not used as a gate.
                        stats["cross_check_iou_sum"] += iou
                        stats["cross_check_iou_n"] += 1
                    elif iou < min_iou:
                        # Painted-ID mode: "everything not background" can swallow
                        # stray meshes, so here the disagreement is a real red flag.
                        stats["rejected_cross_check"] += 1
                        continue

            box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if box_w * box_h < min_area or box_w < 2.0 or box_h < 2.0:
                stats["rejected_small"] += 1
                continue
            if touches_border(bbox, width, height) and box_w * box_h < 4 * min_area:
                stats["rejected_small"] += 1
                continue
            # Final guarantee: the RGB pixels inside the label box must actually
            # show something. Catches any residual seg/scene desync the double
            # capture missed.
            if label_box_looks_empty(frame.rgb, bbox):
                stats["rejected_empty_box"] += 1
                continue

        split = "val" if rnd.random() < val_ratio else "train"
        index = start_index + stats["written"]
        stem = f"frame_{index:06d}"
        cv2.imwrite(str(images_dir / split / f"{stem}.jpg"), frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        label_text = "" if bbox is None else to_yolo_line(bbox, width, height, 0) + "\n"
        (labels_dir / split / f"{stem}.txt").write_text(label_text, encoding="utf-8")

        stats["written"] += 1
        if bbox is None:
            stats["background"] += 1
        if previews_left > 0:
            write_preview(frame.rgb, bbox, out_root / "preview" / f"{stem}.jpg")
            previews_left -= 1
        if stats["written"] % 100 == 0:
            rate = stats["written"] / max(1e-6, time.time() - start_time)
            print(f"written={stats['written']}/{samples_target} attempts={attempts} rate={rate:.1f}/s")

    if args.pause:
        try:
            client.simPause(False)
        except Exception:  # noqa: BLE001
            pass

    data_yaml = out_root / "data.yaml"
    names = cfg["class_names"]
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    stats["elapsed_s"] = round(time.time() - start_time, 1)
    stats["attempts"] = attempts
    stats["data_yaml"] = str(data_yaml)
    if stats["cross_check_iou_n"]:
        stats["cross_check_iou_mean"] = round(stats["cross_check_iou_sum"] / stats["cross_check_iou_n"], 3)
    del stats["cross_check_iou_sum"], stats["cross_check_iou_n"]
    save_json(out_root / "collection_stats.json", stats)
    print(json.dumps(stats, indent=2))
    print(f"data_yaml={data_yaml}")
    if stats["written"] < samples_target:
        print(
            "WARNING: collected fewer samples than requested — inspect the reject counters "
            "and the preview/ folder before training."
        )


if __name__ == "__main__":
    main()
