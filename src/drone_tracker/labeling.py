from __future__ import annotations

import math

import numpy as np

from .utils import focal_px

BBox = tuple[float, float, float, float]


def _pack_colors(image: np.ndarray) -> np.ndarray:
    flat = image.reshape(-1, 3).astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def _border_modal_color(packed: np.ndarray, height: int, width: int, ring: int = 2) -> np.uint32:
    """Background colour, sampled from the image border ring.

    Using the border rather than the global mode keeps this correct even when the
    target fills most of the frame at close range.
    """
    grid = packed.reshape(height, width)
    border = np.concatenate(
        [
            grid[:ring, :].ravel(),
            grid[-ring:, :].ravel(),
            grid[:, :ring].ravel(),
            grid[:, -ring:].ravel(),
        ]
    )
    values, counts = np.unique(border, return_counts=True)
    return values[int(np.argmax(counts))]


def bbox_and_pixels_from_segmentation(
    segmentation: np.ndarray,
    min_area_px: int = 16,
    largest_component_only: bool = True,
    target_colors: list[tuple[int, int, int]] | None = None,
) -> tuple[BBox | None, int]:
    """Like :func:`bbox_from_segmentation` but also returns the mask pixel count.

    The count is the visible target area — comparing it against the expected area
    at the known distance exposes partial occlusion (a sliver of drone peeking past
    a building edge yields a valid but tiny mask).
    """
    bbox = bbox_from_segmentation(
        segmentation,
        min_area_px=min_area_px,
        largest_component_only=largest_component_only,
        target_colors=target_colors,
        _pixel_count_out=(out := [0]),
    )
    return bbox, out[0]


def bbox_from_segmentation(
    segmentation: np.ndarray,
    min_area_px: int = 16,
    largest_component_only: bool = True,
    target_colors: list[tuple[int, int, int]] | None = None,
    _pixel_count_out: list[int] | None = None,
) -> BBox | None:
    """Tightest box around the target's pixels in the segmentation frame.

    Two labelling schemes:
    - ``target_colors`` given (Cosys-AirSim instance segmentation): the mask is
      every pixel exactly matching one of the target mesh colours.
    - ``target_colors`` None (old painted-ID scheme): assumes
      :func:`airsim_io.configure_segmentation` painted the scene with id 0 and only
      the target with a distinct id, so any non-background pixel is the target.
    """
    if segmentation is None or segmentation.ndim != 3:
        return None
    height, width = segmentation.shape[:2]
    packed = _pack_colors(segmentation)
    if target_colors:
        wanted = np.array(
            [(int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in target_colors], dtype=np.uint32
        )
        mask = np.isin(packed, wanted).reshape(height, width)
    else:
        background = _border_modal_color(packed, height, width)
        mask = (packed != background).reshape(height, width)
    pixels = int(mask.sum())
    if _pixel_count_out is not None:
        _pixel_count_out[0] = pixels
    if pixels < min_area_px:
        return None

    if largest_component_only:
        mask = _largest_component(mask)
        if mask is None:
            return None
        pixels = int(mask.sum())
        if _pixel_count_out is not None:
            _pixel_count_out[0] = pixels
        if pixels < min_area_px:
            return None

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return (float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1))


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Drop stray specks (distant scenery painted with a stale id) if cv2 is present."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is an optional hardening step
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    return labels == best


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def project_point(
    world_point: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    fov_deg: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float] | None:
    """Pinhole-project a world point into the AirSim camera.

    AirSim's camera frame is +x forward, +y right, +z down. Returns ``(u, v, depth)``
    or ``None`` when the point is behind the image plane.
    """
    relative = camera_rotation.T @ (np.asarray(world_point, dtype=float) - camera_position)
    forward = float(relative[0])
    if forward <= 1e-6:
        return None
    f = focal_px(fov_deg, image_width)
    u = image_width / 2.0 + f * float(relative[1]) / forward
    v = image_height / 2.0 + f * float(relative[2]) / forward
    return u, v, forward


def bbox_from_projection(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    extents_m: tuple[float, float, float],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    fov_deg: float,
    image_width: int,
    image_height: int,
    clip_to_image: bool = True,
) -> BBox | None:
    """Project the target's 3D bounding cuboid and take the enclosing 2D box.

    Ground-truth fallback for when segmentation ids are unavailable. It is looser
    than the segmentation box (a cuboid over-covers a thin airframe), so prefer
    segmentation when both are available.
    """
    half = np.array(extents_m, dtype=float) / 2.0
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float)
                corners.append(np.asarray(target_position, dtype=float) + target_rotation @ local)

    points = []
    for corner in corners:
        projected = project_point(
            corner, camera_position, camera_rotation, fov_deg, image_width, image_height
        )
        if projected is None:
            return None
        points.append(projected[:2])

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if clip_to_image:
        x1 = max(0.0, min(x1, image_width))
        x2 = max(0.0, min(x2, image_width))
        y1 = max(0.0, min(y1, image_height))
        y2 = max(0.0, min(y2, image_height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def to_yolo_line(bbox: BBox, image_width: int, image_height: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / image_width
    cy = (y1 + y2) / 2.0 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def box_structure_scores(image_bgr: np.ndarray, bbox: BBox, pad_scale: float = 1.0) -> tuple[float, float]:
    """Mean gradient magnitude inside ``bbox`` vs in the ring of background around it.

    A real object in the box produces markedly more edge energy inside than the
    empty sky/ground behind it. Used both by the collector (to refuse writing a
    label whose box contains nothing — segmentation and scene render passes can
    desynchronise on Cosys-AirSim 3.4) and by the offline dataset audit.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return 0.0, 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0, 0.0
    box_w, box_h = x2 - x1, y2 - y1
    rx1 = max(0, int(x1 - box_w * pad_scale))
    ry1 = max(0, int(y1 - box_h * pad_scale))
    rx2 = min(width, int(x2 + box_w * pad_scale))
    ry2 = min(height, int(y2 + box_h * pad_scale))

    region = gray[ry1:ry2, rx1:rx2].astype(np.float32)
    grad = np.sqrt(
        cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3) ** 2
    )
    bx1, by1 = x1 - rx1, y1 - ry1
    bx2, by2 = bx1 + box_w, by1 + box_h
    inside_patch = grad[by1:by2, bx1:bx2]
    inside = float(inside_patch.mean())
    ring_sum = float(grad.sum()) - float(inside_patch.sum())
    ring_area = grad.size - inside_patch.size
    ring = ring_sum / ring_area if ring_area > 0 else 0.0
    return inside, ring


def label_box_looks_empty(
    image_bgr: np.ndarray,
    bbox: BBox,
    min_inside: float = 4.0,
    min_ratio: float = 1.05,
) -> bool:
    """True when the labelled box shows no more structure than its surroundings."""
    inside, ring = box_structure_scores(image_bgr, bbox)
    return inside < min_inside or inside < ring * min_ratio


def degenerate_frame_reason(
    rgb: np.ndarray,
    depth: np.ndarray | None = None,
    min_rgb_std: float = 6.0,
    min_median_depth_m: float = 1.0,
) -> str | None:
    """Reason string if the frame is unusable for training, else None.

    - ``flat_rgb``: near-uniform colour — the wall-interior / featureless capture.
    - ``inside_geometry``: median depth below ``min_median_depth_m`` — the camera
      spawned inside a mesh, everything is centimetres away.
    """
    sub = rgb[::8, ::8].astype(np.float32)
    if float(sub.std()) < min_rgb_std:
        return "flat_rgb"
    if depth is not None and depth.size:
        finite = depth[np.isfinite(depth)]
        finite = finite[finite > 0.0]
        if finite.size and float(np.median(finite)) < min_median_depth_m:
            return "inside_geometry"
    return None


def touches_border(bbox: BBox, image_width: int, image_height: int, margin_px: float = 2.0) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= image_width - margin_px
        or y2 >= image_height - margin_px
    )
