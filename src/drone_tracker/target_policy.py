from __future__ import annotations

import math
from typing import Callable

# All helpers return (x, y, z, yaw_deg) in AirSim world NED coordinates:
#   +x forward, +y right, +z DOWN. Altitude therefore maps to a negative z.


def _phase(seed: int, index: int) -> float:
    """Deterministic pseudo-random phase in [0, 2*pi) — reproducible across runs."""
    value = math.sin(float(seed) * 12.9898 + float(index) * 78.233) * 43758.5453
    return (value - math.floor(value)) * 2.0 * math.pi


def target_position_at(t_s: float, policy: dict) -> tuple[float, float, float, float]:
    pattern = str(policy.get("pattern", "front_sweep"))
    altitude_m = float(policy.get("altitude_m", 8.0))
    speed = float(policy.get("speed_m_s", 0.75))
    radius_m = float(policy.get("radius_m", 4.0))
    base_x = float(policy.get("base_x_m", 2.0))
    forward_amp = float(policy.get("forward_amp_m", 3.0))
    lateral_amp = float(policy.get("lateral_amp_m", 0.9))
    vertical_amp = float(policy.get("vertical_amp_m", 0.2))
    w = speed

    if pattern == "hover":
        x, y, z_up = base_x, 0.0, altitude_m
    elif pattern == "front_sweep":
        x = base_x + forward_amp * math.sin(w * t_s)
        y = lateral_amp * math.sin(2.0 * w * t_s)
        z_up = altitude_m + vertical_amp * math.sin(0.7 * w * t_s)
    elif pattern == "circle":
        x = base_x + radius_m * math.cos(w * t_s)
        y = radius_m * math.sin(w * t_s)
        z_up = altitude_m + vertical_amp * math.sin(0.5 * w * t_s)
    elif pattern == "figure8":
        x = base_x + radius_m * math.sin(w * t_s)
        y = radius_m * math.sin(2.0 * w * t_s) / 2.0
        z_up = altitude_m + vertical_amp * math.sin(1.3 * w * t_s)
    elif pattern == "lateral_dash":
        # Sharp direction reversals — the hardest case for a constant-velocity predictor.
        x = base_x
        y = lateral_amp * math.tanh(4.0 * math.sin(w * t_s))
        z_up = altitude_m
    elif pattern == "random_walk":
        seed = int(policy.get("seed", 7))
        harmonics = int(policy.get("harmonics", 3))
        x = base_x
        y = 0.0
        z_up = altitude_m
        for k in range(1, harmonics + 1):
            wk = w * (0.6 + 0.45 * k)
            x += (forward_amp / k) * math.sin(wk * t_s + _phase(seed, k))
            y += (lateral_amp / k) * math.sin(wk * t_s + _phase(seed, k + 100))
            z_up += (vertical_amp / k) * math.sin(wk * t_s + _phase(seed, k + 200))
    else:
        raise ValueError(f"unknown target_policy.pattern: {pattern!r}")

    yaw_deg = float(policy.get("yaw_deg", 180.0))
    if policy.get("yaw_follows_path", False):
        ahead = 0.05
        nx, ny, _, _ = target_position_at(t_s + ahead, {**policy, "yaw_follows_path": False})
        yaw_deg = math.degrees(math.atan2(ny - y, nx - x))

    return x, y, -abs(z_up), yaw_deg


def move_target(
    client,
    vehicle_name: str,
    elapsed_s: float,
    policy: dict,
    pose_fn: Callable[[float, float, float, float], object],
) -> tuple[float, float, float, float]:
    """Teleport the target along its scripted path.

    Pose-setting (rather than velocity commands) keeps the ground-truth trajectory
    exactly reproducible, which is what makes the A/B comparison meaningful.
    Velocity is zeroed on every step — otherwise gravity accumulates across
    teleports and the target sags visibly between control ticks.

    NOTE: on Cosys-AirSim 3.4 high-rate teleports desynchronise the render from the
    physics state. Prefer :func:`follow_target` for closed-loop episodes there.
    """
    x, y, z, yaw_deg = target_position_at(elapsed_s, policy)
    from .airsim_io import teleport_still

    teleport_still(client, vehicle_name, x, y, z, yaw_deg)
    return x, y, z, yaw_deg


def follow_target(
    client,
    vehicle_name: str,
    elapsed_s: float,
    dt_s: float,
    policy: dict,
) -> tuple[float, float, float, float]:
    """Fly the target along its scripted path with velocity commands.

    Physics-driven motion keeps the render authoritative (no teleport desync) at
    the cost of a small, bounded path-following error. The reference trajectory
    itself stays deterministic, so A/B runs remain comparable.
    """
    ref_x, ref_y, ref_z, yaw_deg = target_position_at(elapsed_s, policy)
    nxt_x, nxt_y, nxt_z, _ = target_position_at(elapsed_s + dt_s, policy)
    position = client.simGetVehiclePose(vehicle_name).position
    vmax = float(policy.get("follow_max_speed_m_s", 6.0))
    # Feed-forward path velocity plus a GENTLE position correction. A stiff 1/dt
    # correction gain turns any tracking error into a full-throttle dash, which
    # makes the target jump around like nothing a real aircraft could fly.
    k_p = float(policy.get("follow_kp", 1.2))
    vx = (nxt_x - ref_x) / dt_s + k_p * (ref_x - position.x_val)
    vy = (nxt_y - ref_y) / dt_s + k_p * (ref_y - position.y_val)
    vz = (nxt_z - ref_z) / dt_s + k_p * (ref_z - position.z_val)
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm > vmax:
        scale = vmax / norm
        vx, vy, vz = vx * scale, vy * scale, vz * scale

    from .airsim_io import airsim_module

    airsim = airsim_module()
    client.moveByVelocityAsync(
        vx,
        vy,
        vz,
        max(2.0 * dt_s, 0.1),
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, float(yaw_deg)),
        vehicle_name=vehicle_name,
    )
    return ref_x, ref_y, ref_z, yaw_deg
