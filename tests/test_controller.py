"""Offline checks on the visual servo control law — no AirSim, no GPU, no weights.

Catches sign flips and clamp regressions, which are the failures that turn into a
drone flying away from its target in the simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import deep_update, load_json  # noqa: E402
from drone_tracker.controller import GimbalState, VisualServoController  # noqa: E402
from drone_tracker.detector import Detection  # noqa: E402
from drone_tracker.prediction import make_predictor, passthrough  # noqa: E402
from drone_tracker.target_policy import target_position_at  # noqa: E402

WIDTH, HEIGHT = 1280, 720


def box_at(cx: float, cy: float, w: float = 120.0, h: float = 75.0) -> Detection:
    return Detection((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.9, 0)


def test_control_signs(cfg: dict) -> None:
    controller = VisualServoController(cfg["control"])

    right = controller.command(box_at(WIDTH * 0.8, HEIGHT / 2), None, WIDTH, HEIGHT, 0.0)
    assert right.yaw_rate_deg_s > 0, "target on the right must yaw right"
    left = controller.command(box_at(WIDTH * 0.2, HEIGHT / 2), None, WIDTH, HEIGHT, 0.0)
    assert left.yaw_rate_deg_s < 0, "target on the left must yaw left"

    low = controller.command(box_at(WIDTH / 2, HEIGHT * 0.85), None, WIDTH, HEIGHT, 0.0)
    assert low.vz > 0, "target low in frame must descend (NED +z is down)"
    high = controller.command(box_at(WIDTH / 2, HEIGHT * 0.15), None, WIDTH, HEIGHT, 0.0)
    assert high.vz < 0, "target high in frame must climb"

    desired_w = cfg["control"]["desired_box_width_norm"] * WIDTH
    far = controller.command(box_at(WIDTH / 2, HEIGHT / 2, w=desired_w * 0.4), None, WIDTH, HEIGHT, 0.0)
    assert far.vx > 0, "small box means far away — close in"
    near = controller.command(box_at(WIDTH / 2, HEIGHT / 2, w=desired_w * 2.0), None, WIDTH, HEIGHT, 0.0)
    assert near.vx < 0, "large box means too close — back off"


def test_deadzone_and_limits(cfg: dict) -> None:
    controller = VisualServoController(cfg["control"])
    deadzone = cfg["control"]["center_deadzone"]

    nudge = (deadzone * 0.5) * (WIDTH / 2)
    centered = controller.command(box_at(WIDTH / 2 + nudge, HEIGHT / 2), None, WIDTH, HEIGHT, 0.0)
    assert centered.yaw_rate_deg_s == 0.0, "inside the deadzone the controller must not chatter"

    extreme = controller.command(box_at(WIDTH - 1, HEIGHT - 1), None, WIDTH, HEIGHT, 0.0)
    assert abs(extreme.yaw_rate_deg_s) <= cfg["control"]["max_yaw_rate_deg_s"] + 1e-9
    assert abs(extreme.vz) <= cfg["control"]["max_vertical_m_s"] + 1e-9
    assert -cfg["control"]["max_reverse_m_s"] - 1e-9 <= extreme.vx <= cfg["control"]["max_forward_m_s"] + 1e-9


def test_lost_behaviour(cfg: dict) -> None:
    controller = VisualServoController(cfg["control"])
    controller.command(box_at(WIDTH * 0.9, HEIGHT / 2), None, WIDTH, HEIGHT, 0.0)

    grace = controller.command(None, None, WIDTH, HEIGHT, cfg["control"]["lost_grace_s"] * 0.5)
    assert grace.yaw_rate_deg_s == 0.0 and not grace.target_visible, "short dropout must hold, not scan"

    scan = controller.command(None, None, WIDTH, HEIGHT, cfg["control"]["lost_grace_s"] * 3.0)
    assert scan.yaw_rate_deg_s > 0, "must scan toward where the target was last seen (right)"
    assert abs(scan.yaw_rate_deg_s) == cfg["control"]["lost_scan_yaw_rate_deg_s"]


def test_depth_distance(cfg: dict) -> None:
    control_cfg = dict(cfg["control"])
    control_cfg["use_depth_distance"] = True
    controller = VisualServoController(control_cfg)

    depth = np.full((HEIGHT, WIDTH), 1000.0, dtype=np.float32)  # AirSim sky
    depth[300:420, 580:700] = 12.0  # the target patch
    command = controller.command(box_at(640, 360, w=120, h=120), depth, WIDTH, HEIGHT, 0.0)
    assert command.distance_m is not None and abs(command.distance_m - 12.0) < 1e-3
    assert command.vx > 0, "12 m away with a 3 m setpoint means close in"


def test_gimbal_follow(cfg: dict) -> None:
    """The airframe must chase the gimbal's line of sight, not hover while it pans."""
    import math

    from drone_tracker.controller import ControlCommand, apply_gimbal_follow

    gimbal_cfg = {"body_follow_gain": 2.0, "max_body_yaw_rate_deg_s": 55.0}

    def fresh() -> ControlCommand:
        return ControlCommand(
            vx=3.0, vy=0.0, vz=0.0, yaw_rate_deg_s=0.0,
            center_error_x=0.0, center_error_y=0.0, distance_m=None, target_visible=True,
        )

    # Gimbal deflected right -> body yaws right and velocity points along the ray.
    right = apply_gimbal_follow(fresh(), 30.0, gimbal_cfg)
    assert right.yaw_rate_deg_s > 0, "body must turn toward the gimbal direction"
    assert abs(right.vx - 3.0 * math.cos(math.radians(30))) < 1e-9
    assert abs(right.vy - 3.0 * math.sin(math.radians(30))) < 1e-9, "pursuit must move along the camera ray"

    left = apply_gimbal_follow(fresh(), -40.0, gimbal_cfg)
    assert left.yaw_rate_deg_s < 0 and left.vy < 0

    # Centered gimbal changes nothing.
    center = apply_gimbal_follow(fresh(), 0.0, gimbal_cfg)
    assert center.yaw_rate_deg_s == 0.0 and center.vy == 0.0 and center.vx == 3.0

    # Yaw rate stays clamped.
    extreme = apply_gimbal_follow(fresh(), 55.0, gimbal_cfg)
    assert abs(extreme.yaw_rate_deg_s) <= gimbal_cfg["max_body_yaw_rate_deg_s"] + 1e-9


def test_gimbal(cfg: dict) -> None:
    gimbal = GimbalState()
    for _ in range(50):
        gimbal.update(1.0, 1.0, cfg["gimbal"])
    assert abs(gimbal.yaw_deg) <= cfg["gimbal"]["max_yaw_deg"] + 1e-9
    assert abs(gimbal.pitch_deg) <= cfg["gimbal"]["max_pitch_deg"] + 1e-9

    gimbal.reset()
    gimbal.scan(1.0, 0.05, {**cfg["gimbal"], "lost_scan_yaw_rate_deg_s": 22.0})
    assert gimbal.yaw_deg > 0


def test_target_policies(cfg: dict) -> None:
    for pattern in ("hover", "front_sweep", "circle", "figure8", "lateral_dash", "random_walk"):
        policy = {**cfg["target_policy"], "pattern": pattern}
        for t in (0.0, 3.7, 41.2):
            x, y, z, yaw = target_position_at(t, policy)
            assert all(np.isfinite([x, y, z, yaw])), (pattern, t)
            assert z < 0, f"{pattern}: NED altitude must be negative, got {z}"
        # Deterministic: same input, same output — required for a meaningful A/B.
        assert target_position_at(5.0, policy) == target_position_at(5.0, policy)


def test_predictor_factory(cfg: dict) -> None:
    assert make_predictor({"enabled": False}) is None
    for model in ("kalman", "imm"):
        predictor = make_predictor(deep_update(cfg["prediction"], {"model": model}))
        assert predictor is not None
        result = predictor.step(box_at(640, 360), 0.0, WIDTH, HEIGHT)
        assert result.detection is not None, model

    empty = passthrough(None)
    assert empty.detection is None and not empty.used_prediction
    held = passthrough(box_at(100, 100))
    assert held.detection is not None and held.raw_center == held.predicted_center


def main() -> None:
    cfg = load_json(ROOT / "config" / "tracking_config.json")
    test_control_signs(cfg)
    test_deadzone_and_limits(cfg)
    test_lost_behaviour(cfg)
    test_depth_distance(cfg)
    test_gimbal_follow(cfg)
    test_gimbal(cfg)
    test_target_policies(cfg)
    test_predictor_factory(cfg)
    print("controller_tests_passed=true")


if __name__ == "__main__":
    main()
