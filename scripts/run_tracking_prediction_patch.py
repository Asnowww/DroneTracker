from __future__ import annotations

"""Reference integration for motion-prediction tracking.

This file shows the exact call order to merge into an existing AirSim
``scripts/run_tracking.py``. It is intentionally small so it can be copied into
the production script without changing the detector or controller APIs.
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.detector import Detection
from drone_tracker.predictor import ImagePlaneKalmanPredictor, PredictionResult


def build_predictor(cfg: dict) -> ImagePlaneKalmanPredictor | None:
    pred_cfg = cfg.get("prediction", {})
    if not pred_cfg.get("enabled", False):
        return None
    return ImagePlaneKalmanPredictor(
        horizon_s=float(pred_cfg.get("horizon_s", 0.2)),
        max_prediction_gap_s=float(pred_cfg.get("max_prediction_gap_s", 0.6)),
        process_noise=float(pred_cfg.get("process_noise", 80.0)),
        measurement_noise=float(pred_cfg.get("measurement_noise", 25.0)),
    )


def predicted_detection_for_control(
    predictor: ImagePlaneKalmanPredictor | None,
    raw_detection: Detection | None,
    timestamp_s: float,
    image_width: int,
    image_height: int,
) -> PredictionResult:
    if predictor is None:
        return PredictionResult(
            detection=raw_detection,
            used_prediction=False,
            predicted_center=raw_detection.center if raw_detection else None,
            raw_center=raw_detection.center if raw_detection else None,
            prediction_age_s=0.0 if raw_detection else None,
        )
    return predictor.step(raw_detection, timestamp_s, image_width, image_height)


def example_loop_fragment(cfg: dict, detector, controller, frame) -> None:
    image_width = int(cfg["image_width"])
    image_height = int(cfg["image_height"])
    predictor = build_predictor(cfg)

    loop_start = time.monotonic()
    raw_detection = detector.detect(frame.rgb)
    prediction = predicted_detection_for_control(
        predictor,
        raw_detection,
        loop_start,
        image_width,
        image_height,
    )
    command = controller.command(
        prediction.detection,
        frame.depth,
        image_width,
        image_height,
        lost_time_s=0.0,
    )
    _ = command

