from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass

import numpy as np

_MODULE = None
_MODULE_NAME = ""


def airsim_module():
    """Import the AirSim python client, preferring Cosys-AirSim.

    Cosys-AirSim ships as ``cosysairsim`` while upstream AirSim/Colosseum ships as
    ``airsim``. The RPC surface used here is identical between them.
    """
    global _MODULE, _MODULE_NAME
    if _MODULE is not None:
        return _MODULE
    errors = []
    for name in ("cosysairsim", "airsim"):
        try:
            _MODULE = importlib.import_module(name)
            _MODULE_NAME = name
            return _MODULE
        except ImportError as exc:  # pragma: no cover - depends on local install
            errors.append(f"{name}: {exc}")
    raise ImportError(
        "Neither 'cosysairsim' nor 'airsim' could be imported.\n"
        + "\n".join(errors)
        + "\nInstall with: pip install airsim   (or: pip install -e <Cosys-AirSim>/PythonClient)"
    )


def module_name() -> str:
    airsim_module()
    return _MODULE_NAME


def _image_type(name: str):
    airsim = airsim_module()
    return getattr(airsim.ImageType, name)


def _depth_image_type():
    """``DepthPlanar`` in Cosys-AirSim, ``DepthPlanner`` in older upstream builds."""
    airsim = airsim_module()
    for name in ("DepthPlanar", "DepthPlanner"):
        if hasattr(airsim.ImageType, name):
            return getattr(airsim.ImageType, name)
    raise AttributeError("AirSim ImageType has no planar depth member")


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None


def connect(ip: str = "", port: int = 41451, timeout_s: float = 30.0):
    """Connect to the AirSim RPC server, retrying until ``timeout_s`` elapses."""
    airsim = airsim_module()
    try:
        client = airsim.MultirotorClient(ip=ip or "127.0.0.1", port=port)
    except TypeError:  # pragma: no cover - very old clients take no port kwarg
        client = airsim.MultirotorClient(ip=ip or "127.0.0.1")

    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.ping()
            return client
        except Exception as exc:  # noqa: BLE001 - rpc layer raises many types
            last_error = exc
            time.sleep(1.0)
    raise ConnectionError(
        f"AirSim RPC not reachable at {ip or '127.0.0.1'}:{port} after {timeout_s:.0f}s. "
        f"Is the simulator running? Last error: {last_error}"
    )


def list_vehicles(client) -> list[str]:
    try:
        return list(client.listVehicles())
    except Exception:  # noqa: BLE001 - older builds lack listVehicles
        return []


def require_vehicles(client, names: list[str]) -> None:
    available = list_vehicles(client)
    if available:
        missing = [name for name in names if name not in available]
        if missing:
            raise RuntimeError(
                f"Vehicles {missing} are not in the simulation (found: {available}). "
                "Check ~/Documents/AirSim/settings.json and restart the simulator."
            )
        return
    # Fall back to probing each vehicle directly.
    for name in names:
        try:
            client.getMultirotorState(vehicle_name=name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Vehicle {name!r} is not available: {exc}") from exc


def arm_and_takeoff(client, names: list[str], altitude_m: float = 8.0, climb_speed_m_s: float = 3.0) -> None:
    for name in names:
        client.enableApiControl(True, name)
        client.armDisarm(True, name)
    for future in [client.takeoffAsync(vehicle_name=name) for name in names]:
        future.join()
    for future in [
        client.moveToZAsync(-abs(altitude_m), climb_speed_m_s, vehicle_name=name) for name in names
    ]:
        future.join()


def hover_lock(client, names: list[str], duration_s: float = 3600.0) -> None:
    """Arm each vehicle and give it a standing zero-velocity command.

    The flight controller then actively fights gravity, so teleports relocate the
    vehicle and it simply hovers at the new spot. Without this, an idle SimpleFlight
    drone free-falls (30 m in ~2.5 s), which desynchronises every capture taken
    moments after a teleport.
    """
    airsim = airsim_module()
    for name in names:
        client.enableApiControl(True, name)
        client.armDisarm(True, name)
    for name in names:
        client.moveByVelocityAsync(
            0.0,
            0.0,
            0.0,
            float(duration_s),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, 0.0),
            vehicle_name=name,
        )


def land_and_disarm(client, names: list[str]) -> None:
    for name in names:
        try:
            client.armDisarm(False, name)
            client.enableApiControl(False, name)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def quaternion_from_euler(pitch_rad: float, roll_rad: float, yaw_rad: float):
    """Euler (NED, pitch-roll-yaw) to an AirSim Quaternionr.

    Same convention as the old ``airsim.to_quaternion`` helper, which Cosys-AirSim
    3.4 removed from the module namespace.
    """
    airsim = airsim_module()
    cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
    cr, sr = math.cos(roll_rad / 2.0), math.sin(roll_rad / 2.0)
    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    return airsim.Quaternionr(
        x_val=sr * cp * cy - cr * sp * sy,
        y_val=cr * sp * cy + sr * cp * sy,
        z_val=cr * cp * sy - sr * sp * cy,
        w_val=cr * cp * cy + sr * sp * sy,
    )


def pose_xyz_yaw(x: float, y: float, z: float, yaw_deg: float):
    """Build an AirSim Pose. ``z`` is NED (negative is up); ``yaw_deg`` is degrees."""
    airsim = airsim_module()
    return airsim.Pose(
        airsim.Vector3r(float(x), float(y), float(z)),
        quaternion_from_euler(0.0, 0.0, math.radians(float(yaw_deg))),
    )


def teleport_still(
    client,
    vehicle_name: str,
    x: float,
    y: float,
    z: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> None:
    """Teleport a vehicle AND zero its velocity.

    ``simSetVehiclePose`` keeps the physics velocity, so a free-falling drone keeps
    accelerating across teleports (measured: 8 m/s accumulated within seconds, i.e.
    >1 m of sag between teleport and capture). Cosys-AirSim's ``simSetKinematics``
    lets us clear the full kinematic state; on servers without it we fall back to a
    plain pose set.
    """
    orientation = quaternion_from_euler(
        math.radians(float(pitch_deg)), math.radians(float(roll_deg)), math.radians(float(yaw_deg))
    )
    try:
        state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
        state.position.x_val = float(x)
        state.position.y_val = float(y)
        state.position.z_val = float(z)
        state.orientation = orientation
        for vec in (
            state.linear_velocity,
            state.angular_velocity,
            state.linear_acceleration,
            state.angular_acceleration,
        ):
            vec.x_val = vec.y_val = vec.z_val = 0.0
        client.simSetKinematics(state, ignore_collision=True, vehicle_name=vehicle_name)
    except Exception:  # noqa: BLE001 - old servers lack simSetKinematics
        airsim = airsim_module()
        pose = airsim.Pose(airsim.Vector3r(float(x), float(y), float(z)), orientation)
        client.simSetVehiclePose(pose, True, vehicle_name)


def _decode_uint8(response) -> np.ndarray | None:
    height, width = int(response.height), int(response.width)
    if height <= 0 or width <= 0:
        return None
    buffer = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if buffer.size < height * width:
        return None
    channels = buffer.size // (height * width)
    image = buffer[: height * width * channels].reshape(height, width, channels)
    if channels >= 4:
        image = image[:, :, :3]
    elif channels == 1:
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image)  # BGR


def _decode_float(response) -> np.ndarray | None:
    height, width = int(response.height), int(response.width)
    if height <= 0 or width <= 0:
        return None
    buffer = np.array(response.image_data_float, dtype=np.float32)
    if buffer.size < height * width:
        return None
    return buffer[: height * width].reshape(height, width)


def _get_images(client, camera_name: str, vehicle_name: str, requests) -> list:
    try:
        return client.simGetImages(requests, vehicle_name=vehicle_name)
    except Exception:  # noqa: BLE001 - transient rpc hiccups should not kill the loop
        return []


def get_scene_and_depth(client, camera_name: str, vehicle_name: str, with_depth: bool = True) -> Frame | None:
    """Scene capture, optionally with planar depth.

    Depth roughly doubles the RPC payload; the tracking loop only requests it when
    ``control.use_depth_distance`` is on.
    """
    airsim = airsim_module()
    requests = [airsim.ImageRequest(camera_name, _image_type("Scene"), False, False)]
    if with_depth:
        requests.append(airsim.ImageRequest(camera_name, _depth_image_type(), True, False))
    responses = _get_images(client, camera_name, vehicle_name, requests)
    if len(responses) < 1:
        return None
    rgb = _decode_uint8(responses[0])
    if rgb is None:
        return None
    depth = _decode_float(responses[1]) if with_depth and len(responses) > 1 else None
    return Frame(rgb=rgb, depth=depth)


def get_scene_and_segmentation(client, camera_name: str, vehicle_name: str) -> Frame | None:
    """Scene + segmentation, used by the dataset collector to auto-label boxes."""
    airsim = airsim_module()
    requests = [
        airsim.ImageRequest(camera_name, _image_type("Scene"), False, False),
        airsim.ImageRequest(camera_name, _image_type("Segmentation"), False, False),
    ]
    responses = _get_images(client, camera_name, vehicle_name, requests)
    if len(responses) < 2:
        return None
    rgb = _decode_uint8(responses[0])
    seg = _decode_uint8(responses[1])
    if rgb is None or seg is None:
        return None
    return Frame(rgb=rgb, segmentation=seg)


def get_scene_depth_and_segmentation(client, camera_name: str, vehicle_name: str) -> Frame | None:
    """Scene + depth + segmentation in one same-frame request.

    Depth is used by the collector to detect a camera stuck inside geometry
    (uniformly tiny depth) before a frame can reach the dataset.
    """
    airsim = airsim_module()
    requests = [
        airsim.ImageRequest(camera_name, _image_type("Scene"), False, False),
        airsim.ImageRequest(camera_name, _depth_image_type(), True, False),
        airsim.ImageRequest(camera_name, _image_type("Segmentation"), False, False),
    ]
    responses = _get_images(client, camera_name, vehicle_name, requests)
    if len(responses) < 3:
        return None
    rgb = _decode_uint8(responses[0])
    depth = _decode_float(responses[1])
    seg = _decode_uint8(responses[2])
    if rgb is None or seg is None:
        return None
    return Frame(rgb=rgb, depth=depth, segmentation=seg)


def line_of_sight_clear(client, point_a, point_b) -> bool | None:
    """True if nothing blocks the segment between two world points.

    Returns None when the server lacks the API (treat as unknown, don't reject).
    """
    airsim = airsim_module()
    try:
        a = airsim.Vector3r(float(point_a[0]), float(point_a[1]), float(point_a[2]))
        b = airsim.Vector3r(float(point_b[0]), float(point_b[1]), float(point_b[2]))
        return bool(client.simTestLineOfSightBetweenPoints(a, b))
    except Exception:  # noqa: BLE001 - API missing on old servers
        return None


def move_body_velocity(
    client,
    vehicle_name: str,
    vx: float,
    vy: float,
    vz: float,
    yaw_rate_deg_s: float,
    duration_s: float,
) -> None:
    """Fire-and-forget body-frame velocity command with a yaw rate.

    Not joined on purpose: the control loop re-issues at ``control_hz`` and blocking
    here would cap the achievable loop rate.
    """
    airsim = airsim_module()
    client.moveByVelocityBodyFrameAsync(
        float(vx),
        float(vy),
        float(vz),
        float(duration_s),
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(True, float(yaw_rate_deg_s)),
        vehicle_name=vehicle_name,
    )


def set_camera_gimbal(client, vehicle_name: str, camera_name: str, yaw_deg: float, pitch_deg: float) -> None:
    airsim = airsim_module()
    pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        quaternion_from_euler(math.radians(float(pitch_deg)), 0.0, math.radians(float(yaw_deg))),
    )
    client.simSetCameraPose(camera_name, pose, vehicle_name=vehicle_name)


def get_camera_info(client, camera_name: str, vehicle_name: str):
    return client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)


def vehicle_pitch_deg(client, vehicle_name: str) -> float:
    """Current body pitch in degrees (positive = nose up)."""
    o = client.simGetVehiclePose(vehicle_name).orientation
    sinp = 2.0 * (o.w_val * o.y_val - o.z_val * o.x_val)
    sinp = max(-1.0, min(1.0, sinp))
    return math.degrees(math.asin(sinp))


def configure_segmentation(client, target_regex: str, target_id: int = 25) -> bool:
    """Paint the whole scene with segmentation id 0, then the target mesh with ``target_id``.

    Returns whether the target regex matched anything. With everything else at id 0,
    any pixel whose colour differs from the modal colour belongs to the target — so
    the collector never needs AirSim's colour palette.

    Old-style (Microsoft AirSim / Colosseum) API. On Cosys-AirSim 3.4+ these calls
    return False — use :func:`get_target_instance_colors` there instead.
    """
    client.simSetSegmentationObjectID("[\\w]*", 0, True)
    time.sleep(0.1)
    return bool(client.simSetSegmentationObjectID(target_regex, int(target_id), True))


def get_target_instance_colors(client, mesh_regex: str) -> list[tuple[int, int, int]] | None:
    """Look up the target's instance-segmentation colours on Cosys-AirSim 3.4+.

    Cosys replaced paintable segmentation ids with instance segmentation: every mesh
    gets a fixed unique colour at startup. ``simListInstanceSegmentationObjects()``
    lists meshes in colormap order, so row ``i`` of ``simGetSegmentationColorMap()``
    is mesh ``i``'s colour.

    Returns colour tuples (colormap channel order) for every mesh matching
    ``mesh_regex``, ``[]`` if nothing matched, or ``None`` when the instance API is
    unavailable (old AirSim — fall back to :func:`configure_segmentation`).
    """
    import re

    try:
        objects = list(client.simListInstanceSegmentationObjects())
    except Exception:  # noqa: BLE001 - method absent on old clients/servers
        return None
    if not objects:
        return None
    pattern = re.compile(mesh_regex)
    indices = [i for i, name in enumerate(objects) if pattern.search(name)]
    if not indices:
        return []
    cmap = np.asarray(client.simGetSegmentationColorMap())
    return [tuple(int(v) for v in cmap[i][:3]) for i in indices]


def list_instance_segmentation_objects(client) -> list[str]:
    try:
        return list(client.simListInstanceSegmentationObjects())
    except Exception:  # noqa: BLE001
        return []


def list_scene_objects(client, name_regex: str = ".*") -> list[str]:
    try:
        return list(client.simListSceneObjects(name_regex))
    except Exception:  # noqa: BLE001
        return []


def set_weather(client, enabled: bool = True, **params: float) -> None:
    """Randomise weather for domain variety. Keys: Rain, Snow, Fog, Dust, MapleLeaf, ..."""
    airsim = airsim_module()
    client.simEnableWeather(bool(enabled))
    if not enabled:
        return
    for key, value in params.items():
        param = getattr(airsim.WeatherParameter, key, None)
        if param is not None:
            client.simSetWeatherParameter(param, float(value))


def set_time_of_day(client, enabled: bool, datetime_str: str = "2024-06-21 12:00:00") -> None:
    try:
        client.simSetTimeOfDay(bool(enabled), start_datetime=datetime_str, is_start_datetime_dst=False)
    except Exception:  # noqa: BLE001 - not every environment supports ToD
        pass
