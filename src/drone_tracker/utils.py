from __future__ import annotations

import math
import time


def now_s() -> float:
    """Monotonic wall clock used for control loop timing and Kalman timestamps."""
    return time.perf_counter()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def deadzone(value: float, width: float) -> float:
    """Zero out |value| < width so the controller does not chatter near center."""
    return 0.0 if abs(value) < width else value


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def focal_px(fov_deg: float, image_width: int) -> float:
    """Pinhole focal length in pixels from AirSim's horizontal FOV."""
    return (image_width / 2.0) / math.tan(deg2rad(fov_deg) / 2.0)
