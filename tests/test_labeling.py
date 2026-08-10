"""Offline checks on the auto-labelling maths — no AirSim needed.

The dataset collector is the one component whose bugs are silent: a wrong box just
produces a quietly bad model. These tests pin the segmentation extraction and the
pinhole projection against synthetic frames with known answers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.labeling import (  # noqa: E402
    bbox_from_projection,
    bbox_from_segmentation,
    bbox_iou,
    project_point,
    quaternion_to_matrix,
    to_yolo_line,
    touches_border,
)
from drone_tracker.utils import focal_px  # noqa: E402

WIDTH, HEIGHT = 1280, 720
FOV = 35.0

BACKGROUND = (10, 20, 30)
TARGET = (25, 60, 90)


def make_segmentation(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND
    image[y1:y2, x1:x2] = TARGET
    return image


def test_segmentation_bbox() -> None:
    seg = make_segmentation(500, 300, 620, 375)
    bbox = bbox_from_segmentation(seg)
    assert bbox == (500.0, 300.0, 620.0, 375.0), bbox

    # A target filling most of the frame must not be mistaken for the background.
    big = make_segmentation(40, 30, WIDTH - 40, HEIGHT - 30)
    bbox_big = bbox_from_segmentation(big)
    assert bbox_big == (40.0, 30.0, float(WIDTH - 40), float(HEIGHT - 30)), bbox_big

    # An empty frame yields no box rather than a bogus one.
    assert bbox_from_segmentation(make_segmentation(0, 0, 0, 0)) is None

    # A speck below the area floor is rejected.
    assert bbox_from_segmentation(make_segmentation(100, 100, 102, 102), min_area_px=64) is None


def test_segmentation_ignores_specks() -> None:
    """With cv2 present, a stray mislabelled pixel must not inflate the box."""
    seg = make_segmentation(600, 340, 680, 390)
    seg[10, 1200] = TARGET  # distant speck
    bbox = bbox_from_segmentation(seg, min_area_px=16, largest_component_only=True)
    assert bbox == (600.0, 340.0, 680.0, 390.0), bbox


def test_projection_center() -> None:
    cam_pos = np.zeros(3)
    cam_rot = np.eye(3)

    straight = project_point(np.array([10.0, 0.0, 0.0]), cam_pos, cam_rot, FOV, WIDTH, HEIGHT)
    assert straight is not None
    u, v, depth = straight
    assert abs(u - WIDTH / 2) < 1e-6 and abs(v - HEIGHT / 2) < 1e-6
    assert abs(depth - 10.0) < 1e-9

    f = focal_px(FOV, WIDTH)
    offset = project_point(np.array([10.0, 1.75, 0.0]), cam_pos, cam_rot, FOV, WIDTH, HEIGHT)
    assert offset is not None
    assert abs(offset[0] - (WIDTH / 2 + f * 0.175)) < 1e-6

    # +z is down in the AirSim camera frame.
    below = project_point(np.array([10.0, 0.0, 1.0]), cam_pos, cam_rot, FOV, WIDTH, HEIGHT)
    assert below is not None and below[1] > HEIGHT / 2

    # Behind the image plane -> no projection.
    assert project_point(np.array([-5.0, 0.0, 0.0]), cam_pos, cam_rot, FOV, WIDTH, HEIGHT) is None


def test_projection_scales_with_distance() -> None:
    cam_pos, cam_rot = np.zeros(3), np.eye(3)
    extents = (1.0, 1.0, 0.35)
    near = bbox_from_projection(
        np.array([8.0, 0.0, 0.0]), np.eye(3), extents, cam_pos, cam_rot, FOV, WIDTH, HEIGHT
    )
    far = bbox_from_projection(
        np.array([32.0, 0.0, 0.0]), np.eye(3), extents, cam_pos, cam_rot, FOV, WIDTH, HEIGHT
    )
    assert near is not None and far is not None
    near_w = near[2] - near[0]
    far_w = far[2] - far[0]
    # 4x the distance -> roughly 1/4 the apparent width.
    assert abs(near_w / far_w - 4.0) < 0.35, (near_w, far_w)


def test_yaw_rotation() -> None:
    """A camera yawed +90 deg sees a point that lies to its former right, straight ahead."""
    yaw = math.radians(90.0)
    cam_rot = quaternion_to_matrix(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    projected = project_point(np.array([0.0, 10.0, 0.0]), np.zeros(3), cam_rot, FOV, WIDTH, HEIGHT)
    assert projected is not None
    assert abs(projected[0] - WIDTH / 2) < 1e-6, projected
    assert abs(projected[2] - 10.0) < 1e-6


def test_yolo_line() -> None:
    line = to_yolo_line((500.0, 300.0, 620.0, 380.0), WIDTH, HEIGHT, 0)
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, w, h = (float(p) for p in parts[1:])
    assert abs(cx - 560.0 / WIDTH) < 1e-6
    assert abs(cy - 340.0 / HEIGHT) < 1e-6
    assert abs(w - 120.0 / WIDTH) < 1e-6
    assert abs(h - 80.0 / HEIGHT) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))


def test_iou_and_border() -> None:
    assert abs(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9

    assert touches_border((0.0, 100.0, 50.0, 200.0), WIDTH, HEIGHT)
    assert touches_border((100.0, 100.0, float(WIDTH), 200.0), WIDTH, HEIGHT)
    assert not touches_border((100.0, 100.0, 200.0, 200.0), WIDTH, HEIGHT)


def test_segmentation_matches_projection() -> None:
    """The two labelling paths must agree — that agreement is the collector's gate."""
    cam_pos, cam_rot = np.zeros(3), np.eye(3)
    extents = (1.0, 1.0, 0.35)
    target = np.array([14.0, 0.6, -0.4])
    projected = bbox_from_projection(
        target, np.eye(3), extents, cam_pos, cam_rot, FOV, WIDTH, HEIGHT
    )
    assert projected is not None

    x1, y1, x2, y2 = (int(round(v)) for v in projected)
    seg_bbox = bbox_from_segmentation(make_segmentation(x1, y1, x2, y2))
    assert seg_bbox is not None
    assert bbox_iou(seg_bbox, projected) > 0.95, (seg_bbox, projected)


def test_quality_gates() -> None:
    """Collector gates: empty-box structure check and degenerate-frame detection."""
    from drone_tracker.labeling import degenerate_frame_reason, label_box_looks_empty

    flat = np.full((720, 1280, 3), 120, dtype=np.uint8)
    assert label_box_looks_empty(flat, (600, 300, 700, 360)), "flat crop must read as empty"

    gradient = np.tile(np.linspace(60, 200, 1280).astype(np.uint8), (720, 1))
    scene = np.stack([gradient] * 3, axis=2).copy()
    rng = np.random.default_rng(0)
    scene[310:350, 620:680] = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    assert not label_box_looks_empty(scene, (600, 300, 700, 360)), "textured crop must pass"

    assert degenerate_frame_reason(flat) == "flat_rgb"
    near = np.full((720, 1280), 0.3, dtype=np.float32)
    assert degenerate_frame_reason(scene, near) == "inside_geometry"
    far = np.full((720, 1280), 20.0, dtype=np.float32)
    assert degenerate_frame_reason(scene, far) is None


def main() -> None:
    test_segmentation_bbox()
    test_segmentation_ignores_specks()
    test_projection_center()
    test_projection_scales_with_distance()
    test_yaw_rotation()
    test_yolo_line()
    test_iou_and_border()
    test_segmentation_matches_projection()
    test_quality_gates()
    print("labeling_tests_passed=true")


if __name__ == "__main__":
    main()
