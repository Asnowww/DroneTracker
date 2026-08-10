from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.airsim_io import (
    arm_and_takeoff,
    connect,
    get_scene_and_depth,
    hover_lock,
    move_body_velocity,
    pose_xyz_yaw,
    require_vehicles,
    set_camera_gimbal,
    teleport_still,
    vehicle_pitch_deg,
)
from drone_tracker.config import load_json
from drone_tracker.controller import GimbalState, VisualServoController, apply_gimbal_follow
from drone_tracker.detector import YoloDroneDetector
from drone_tracker.metrics import TrackingSample, summarize_tracking, write_summary, write_tracking_csv
from drone_tracker.prediction import Predictor, make_predictor, passthrough
from drone_tracker.predictor import PredictionResult
from drone_tracker.target_policy import follow_target, target_position_at
from drone_tracker.utils import now_s


def build_predictor(cfg: dict) -> Predictor | None:
    return make_predictor(cfg.get("prediction", {}))


def prediction_for_control(
    predictor: Predictor | None,
    raw_detection,
    timestamp_s: float,
    image_width: int,
    image_height: int,
) -> PredictionResult:
    if predictor is None:
        return passthrough(raw_detection)
    return predictor.step(raw_detection, timestamp_s, image_width, image_height)


def run_episode(cfg: dict, weights: str, seconds: float | None = None) -> dict:
    tracker = cfg["tracker_vehicle"]
    target = cfg["target_vehicle"]
    camera = cfg["camera_name"]
    image_width = int(cfg["image_width"])
    image_height = int(cfg["image_height"])
    detector_cfg = cfg["detector"]
    control_cfg = cfg["control"]
    episode_cfg = cfg["episode"]
    duration_s = float(seconds if seconds is not None else episode_cfg["seconds"])
    dt = 1.0 / float(episode_cfg["control_hz"])

    client = connect()
    require_vehicles(client, [tracker, target])
    target_policy = cfg["target_policy"]
    altitude_m = float(target_policy.get("altitude_m", 8.0))
    desired_distance_m = float(control_cfg["desired_distance_m"])
    arm_and_takeoff(client, [tracker, target], altitude_m=altitude_m)
    # Zero-velocity standing commands: teleported drones hover at the new spot
    # instead of flying back toward a stale position setpoint or free-falling.
    hover_lock(client, [tracker, target])
    client.simSetCameraFov(camera, float(cfg["camera_fov_deg"]), vehicle_name=tracker)
    teleport_still(client, tracker, -desired_distance_m, 0.0, -altitude_m, 0.0)
    target_x, target_y, target_z, target_yaw = target_position_at(0.0, target_policy)
    teleport_still(client, target, target_x, target_y, target_z, target_yaw)
    set_camera_gimbal(client, tracker, camera, 0.0, 0.0)
    time.sleep(1.0)

    detector = YoloDroneDetector(weights, **detector_cfg)
    predictor = build_predictor(cfg)
    controller = VisualServoController(control_cfg)
    gimbal = GimbalState()
    # Slew-limit the horizontal velocity commands: abrupt accelerations tilt the
    # airframe, and a body camera drags the target across the frame with every
    # tilt faster than the pitch compensation can follow.
    max_accel = float(control_cfg.get("max_horizontal_accel_m_s2", 1.5))
    prev_vx, prev_vy = 0.0, 0.0
    prev_loop = now_s()
    samples: list[TrackingSample] = []
    last_seen = now_s()

    use_depth = bool(control_cfg.get("use_depth_distance", False))

    time.sleep(0.5)
    warmup_frame = get_scene_and_depth(client, camera, tracker, with_depth=use_depth)
    if warmup_frame is not None:
        warmup_detection = detector.detect(warmup_frame.rgb)
        if predictor is not None:
            predictor.step(warmup_detection, now_s(), image_width, image_height)

    start = now_s()
    while now_s() - start < duration_s:
        loop_start = now_s()
        elapsed = loop_start - start
        follow_target(client, target, elapsed, dt, cfg["target_policy"])

        frame = get_scene_and_depth(client, camera, tracker, with_depth=use_depth)
        if frame is None:
            time.sleep(dt)
            continue

        raw_detection = detector.detect(frame.rgb)
        prediction = prediction_for_control(predictor, raw_detection, loop_start, image_width, image_height)
        detection_for_control = prediction.detection
        lost_time = loop_start - last_seen
        command = controller.command(detection_for_control, frame.depth, image_width, image_height, lost_time)

        gimbal_cfg = cfg.get("gimbal", {})
        # A body-mounted camera tilts with the airframe: braking pitches the nose
        # up and the target dives toward the frame edge. Mimic a stabilized gimbal
        # by counter-rotating the camera pitch against the measured body pitch.
        pitch_comp = 0.0
        if gimbal_cfg.get("enabled", False) and gimbal_cfg.get("stabilize_pitch", True):
            pitch_comp = -vehicle_pitch_deg(client, tracker)

        if detection_for_control is not None:
            last_seen = loop_start
            if gimbal_cfg.get("enabled", False):
                gimbal.update(command.center_error_x, command.center_error_y, gimbal_cfg)
                set_camera_gimbal(client, tracker, camera, gimbal.yaw_deg, gimbal.pitch_deg + pitch_comp)
                # The airframe chases the gimbal's line of sight — otherwise only
                # the camera tracks while the body hovers in place.
                command = apply_gimbal_follow(command, gimbal.yaw_deg, gimbal_cfg)
        elif gimbal_cfg.get("enabled", False):
            gimbal_scan_cfg = dict(gimbal_cfg)
            gimbal_scan_cfg["lost_scan_yaw_rate_deg_s"] = control_cfg["lost_scan_yaw_rate_deg_s"]
            gimbal.scan(controller.last_seen_direction, dt, gimbal_scan_cfg)
            set_camera_gimbal(client, tracker, camera, gimbal.yaw_deg, gimbal.pitch_deg + pitch_comp)
            # The gimbal only covers +/-max_yaw_deg. Once it saturates, keep the
            # sweep going with the body so a target anywhere on the circle can be
            # reacquired.
            max_gimbal_yaw = float(gimbal_cfg.get("max_yaw_deg", 55.0))
            if abs(gimbal.yaw_deg) >= 0.9 * max_gimbal_yaw:
                command.yaw_rate_deg_s = (
                    controller.last_seen_direction * float(control_cfg["lost_scan_yaw_rate_deg_s"])
                )
            else:
                command.yaw_rate_deg_s = 0.0

        loop_dt = max(1e-3, loop_start - prev_loop)
        prev_loop = loop_start
        max_dv = max_accel * loop_dt
        command.vx = prev_vx + max(-max_dv, min(max_dv, command.vx - prev_vx))
        command.vy = prev_vy + max(-max_dv, min(max_dv, command.vy - prev_vy))
        prev_vx, prev_vy = command.vx, command.vy

        move_body_velocity(
            client,
            tracker,
            command.vx,
            command.vy,
            command.vz,
            command.yaw_rate_deg_s,
            float(control_cfg["command_duration_s"]),
        )

        center_error_px = (
            (command.center_error_x * image_width / 2.0) ** 2
            + (command.center_error_y * image_height / 2.0) ** 2
        ) ** 0.5
        predicted_center = prediction.predicted_center
        raw_center = prediction.raw_center
        samples.append(
            TrackingSample(
                t_s=elapsed,
                visible=command.target_visible,
                center_error_x=command.center_error_x,
                center_error_y=command.center_error_y,
                center_error_px=center_error_px,
                distance_m=command.distance_m,
                vx=command.vx,
                vz=command.vz,
                yaw_rate_deg_s=command.yaw_rate_deg_s,
                confidence=raw_detection.confidence if raw_detection else None,
                prediction_used=prediction.used_prediction,
                predicted_center_x=predicted_center[0] if predicted_center else None,
                predicted_center_y=predicted_center[1] if predicted_center else None,
                raw_center_x=raw_center[0] if raw_center else None,
                raw_center_y=raw_center[1] if raw_center else None,
                prediction_age_s=prediction.prediction_age_s,
            )
        )

        remaining = dt - (now_s() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

    run_dir = Path(cfg["run_dir"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = run_dir / f"tracking_{stamp}.csv"
    summary_path = run_dir / f"tracking_{stamp}.json"
    write_tracking_csv(csv_path, samples)
    summary = summarize_tracking(samples)
    summary["csv"] = str(csv_path)
    summary["weights"] = str(weights)
    pred_cfg = cfg.get("prediction", {})
    summary["prediction_enabled"] = bool(pred_cfg.get("enabled", False))
    summary["prediction_model"] = str(pred_cfg.get("model", "kalman")) if pred_cfg.get("enabled") else "none"
    summary["prediction_horizon_s"] = float(pred_cfg.get("horizon_s", 0.0))
    summary["target_pattern"] = str(cfg["target_policy"].get("pattern", ""))
    summary["episode_seconds"] = duration_s
    write_summary(summary_path, summary)
    print(f"tracking_csv={csv_path}")
    print(f"tracking_summary={summary_path}")
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "tracking_config.json"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()
    cfg = load_json(args.config)
    run_episode(cfg, args.weights, args.seconds)


if __name__ == "__main__":
    main()

