from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detector import Detection
from .utils import clamp, deadzone


@dataclass
class ControlCommand:
    """One control-loop output, in the Tracker body frame (NED: +z is down)."""

    vx: float
    vy: float
    vz: float
    yaw_rate_deg_s: float
    center_error_x: float
    center_error_y: float
    distance_m: float | None
    target_visible: bool


class VisualServoController:
    """Image-plane visual servo: drives the detection box back to frame center.

    Horizontal error -> yaw rate. Vertical error -> body z velocity.
    Box width (or depth median) -> forward/backward velocity.
    """

    def __init__(self, cfg: dict) -> None:
        self.desired_distance_m = float(cfg["desired_distance_m"])
        self.desired_box_width_norm = float(cfg["desired_box_width_norm"])
        self.k_size = float(cfg["k_size"])
        self.use_depth_distance = bool(cfg.get("use_depth_distance", False))
        self.center_deadzone = float(cfg["center_deadzone"])
        self.k_yaw = float(cfg["k_yaw"])
        self.k_z = float(cfg["k_z"])
        self.k_dist = float(cfg["k_dist"])
        self.max_yaw_rate_deg_s = float(cfg["max_yaw_rate_deg_s"])
        self.max_forward_m_s = float(cfg["max_forward_m_s"])
        self.max_reverse_m_s = float(cfg["max_reverse_m_s"])
        self.max_vertical_m_s = float(cfg["max_vertical_m_s"])
        self.lost_grace_s = float(cfg["lost_grace_s"])
        self.lost_scan_yaw_rate_deg_s = float(cfg["lost_scan_yaw_rate_deg_s"])
        # +1 means the target was last seen on the right, -1 on the left.
        self.last_seen_direction: float = 1.0

    def command(
        self,
        detection: Detection | None,
        depth: np.ndarray | None,
        image_width: int,
        image_height: int,
        lost_time_s: float,
    ) -> ControlCommand:
        if detection is None:
            return self._lost_command(lost_time_s)

        cx, cy = detection.center
        box_w, _box_h = detection.size

        error_x = (cx - image_width / 2.0) / (image_width / 2.0)
        error_y = (cy - image_height / 2.0) / (image_height / 2.0)
        if error_x != 0.0:
            self.last_seen_direction = 1.0 if error_x > 0 else -1.0

        servo_x = deadzone(error_x, self.center_deadzone)
        servo_y = deadzone(error_y, self.center_deadzone)

        yaw_rate = clamp(self.k_yaw * servo_x, -self.max_yaw_rate_deg_s, self.max_yaw_rate_deg_s)
        # NED body frame: +z is down. Target below center (servo_y > 0) -> descend.
        vz = clamp(self.k_z * servo_y, -self.max_vertical_m_s, self.max_vertical_m_s)

        distance_m = self._distance_from_depth(detection, depth) if self.use_depth_distance else None
        if distance_m is not None:
            range_error = distance_m - self.desired_distance_m
            vx = clamp(self.k_dist * range_error, -self.max_reverse_m_s, self.max_forward_m_s)
        else:
            # Box too narrow -> target is far -> move forward.
            size_error = self.desired_box_width_norm - (box_w / image_width)
            vx = clamp(self.k_size * size_error, -self.max_reverse_m_s, self.max_forward_m_s)

        return ControlCommand(
            vx=vx,
            vy=0.0,
            vz=vz,
            yaw_rate_deg_s=yaw_rate,
            center_error_x=error_x,
            center_error_y=error_y,
            distance_m=distance_m,
            target_visible=True,
        )

    def _lost_command(self, lost_time_s: float) -> ControlCommand:
        if lost_time_s < self.lost_grace_s:
            # Brief dropout: hold attitude and wait rather than lurching into a scan.
            yaw_rate = 0.0
        else:
            yaw_rate = self.lost_scan_yaw_rate_deg_s * self.last_seen_direction
        return ControlCommand(
            vx=0.0,
            vy=0.0,
            vz=0.0,
            yaw_rate_deg_s=yaw_rate,
            center_error_x=0.0,
            center_error_y=0.0,
            distance_m=None,
            target_visible=False,
        )

    def _distance_from_depth(self, detection: Detection, depth: np.ndarray | None) -> float | None:
        if depth is None or depth.size == 0:
            return None
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = detection.xyxy
        # Sample the inner half of the box: the rim is mostly background sky.
        cx, cy = detection.center
        half_w = max(1.0, (x2 - x1) / 4.0)
        half_h = max(1.0, (y2 - y1) / 4.0)
        c0 = int(max(0, min(width - 1, cx - half_w)))
        c1 = int(max(c0 + 1, min(width, cx + half_w)))
        r0 = int(max(0, min(height - 1, cy - half_h)))
        r1 = int(max(r0 + 1, min(height, cy + half_h)))
        patch = depth[r0:r1, c0:c1]
        finite = patch[np.isfinite(patch)]
        # AirSim reports unreachable sky as a very large planar depth.
        finite = finite[(finite > 0.1) & (finite < 500.0)]
        if finite.size == 0:
            return None
        return float(np.median(finite))


def apply_gimbal_follow(command: ControlCommand, gimbal_yaw_deg: float, cfg: dict) -> ControlCommand:
    """Make the BODY pursue where the gimbal points, not just the camera.

    A gimbal that keeps the target centred also zeroes the image error the body
    yaw controller feeds on — the aircraft then hovers while only the camera
    tracks. Two corrections restore physical pursuit:

    - body yaw rate gets a term proportional to the gimbal deflection, steering
      the airframe toward the line of sight (which lets the gimbal unwind back
      to centre);
    - the range-control velocity is rotated from camera-ray direction into the
      body frame, so "close the distance" pushes toward the target instead of
      toward wherever the nose happens to point.
    """
    follow_gain = float(cfg.get("body_follow_gain", 2.0))  # deg/s of body yaw per deg of gimbal yaw
    max_yaw_rate = float(cfg.get("max_body_yaw_rate_deg_s", 55.0))
    command.yaw_rate_deg_s = clamp(
        command.yaw_rate_deg_s + follow_gain * gimbal_yaw_deg, -max_yaw_rate, max_yaw_rate
    )
    ray = math.radians(gimbal_yaw_deg)
    forward = command.vx
    command.vx = forward * math.cos(ray)
    command.vy = forward * math.sin(ray)
    return command


class GimbalState:
    """Integrating camera gimbal that keeps the target centered without yawing the body."""

    def __init__(self) -> None:
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0

    def update(self, error_x: float, error_y: float, cfg: dict) -> None:
        max_yaw = float(cfg.get("max_yaw_deg", 55.0))
        max_pitch = float(cfg.get("max_pitch_deg", 30.0))
        self.yaw_deg = clamp(self.yaw_deg + float(cfg.get("k_yaw_deg", 0.0)) * error_x, -max_yaw, max_yaw)
        # k_pitch_deg is 0.0 in the tuned config: vertical is handled by body altitude
        # instead, which avoids pitch integrator drift over long runs.
        self.pitch_deg = clamp(
            self.pitch_deg - float(cfg.get("k_pitch_deg", 0.0)) * error_y, -max_pitch, max_pitch
        )

    def scan(self, direction: float, dt: float, cfg: dict) -> None:
        max_yaw = float(cfg.get("max_yaw_deg", 55.0))
        rate = float(cfg.get("lost_scan_yaw_rate_deg_s", 22.0))
        self.yaw_deg = clamp(self.yaw_deg + direction * rate * dt, -max_yaw, max_yaw)
        # Decay the pitch integrator toward level while searching — otherwise the
        # camera sweeps a full circle while staring at the sky or the ground at
        # whatever pitch the integrator held when the target was lost.
        self.pitch_deg *= max(0.0, 1.0 - 2.0 * dt)

    def reset(self) -> None:
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
