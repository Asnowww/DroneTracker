from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Detection


@dataclass
class PredictionResult:
    detection: Detection | None
    used_prediction: bool
    predicted_center: tuple[float, float] | None
    raw_center: tuple[float, float] | None
    prediction_age_s: float | None


class ImagePlaneKalmanPredictor:
    """Constant-velocity Kalman predictor for image-plane YOLO boxes.

    State vector:
        [cx, cy, w, h, vx, vy, vw, vh]

    Measurements:
        [cx, cy, w, h]
    """

    def __init__(
        self,
        horizon_s: float = 0.2,
        max_prediction_gap_s: float = 0.6,
        process_noise: float = 80.0,
        measurement_noise: float = 25.0,
    ) -> None:
        self.horizon_s = float(horizon_s)
        self.max_prediction_gap_s = float(max_prediction_gap_s)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self._x: np.ndarray | None = None
        self._p: np.ndarray | None = None
        self._last_timestamp_s: float | None = None
        self._last_measurement_s: float | None = None
        self._last_confidence = 0.0
        self._last_class_id = 0

    @property
    def initialized(self) -> bool:
        return self._x is not None and self._p is not None

    def reset(self) -> None:
        self._x = None
        self._p = None
        self._last_timestamp_s = None
        self._last_measurement_s = None
        self._last_confidence = 0.0
        self._last_class_id = 0

    def step(
        self,
        detection: Detection | None,
        timestamp_s: float,
        image_width: int,
        image_height: int,
    ) -> PredictionResult:
        raw_center = detection.center if detection is not None else None

        if detection is not None and not self.initialized:
            self._initialize(detection, timestamp_s)
        elif self.initialized:
            dt = max(0.0, float(timestamp_s) - float(self._last_timestamp_s or timestamp_s))
            self._predict_in_place(dt)
            self._last_timestamp_s = float(timestamp_s)
            if detection is not None:
                self._update(detection)
                self._last_measurement_s = float(timestamp_s)
                self._last_confidence = detection.confidence
                self._last_class_id = detection.class_id

        if not self.initialized:
            return PredictionResult(None, False, None, raw_center, None)

        assert self._x is not None
        age_s = float(timestamp_s) - float(self._last_measurement_s or timestamp_s)
        if detection is None and age_s > self.max_prediction_gap_s:
            return PredictionResult(None, False, None, raw_center, age_s)

        future_state = self._predict_state(self._x, self.horizon_s)
        predicted = self._state_to_detection(future_state, image_width, image_height)
        used_prediction = detection is None or self.horizon_s > 0.0
        return PredictionResult(
            detection=predicted,
            used_prediction=used_prediction,
            predicted_center=predicted.center,
            raw_center=raw_center,
            prediction_age_s=age_s,
        )

    def _initialize(self, detection: Detection, timestamp_s: float) -> None:
        cx, cy = detection.center
        w, h = detection.size
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._p = np.diag([100.0, 100.0, 100.0, 100.0, 500.0, 500.0, 500.0, 500.0])
        self._last_timestamp_s = float(timestamp_s)
        self._last_measurement_s = float(timestamp_s)
        self._last_confidence = detection.confidence
        self._last_class_id = detection.class_id

    def _transition(self, dt: float) -> np.ndarray:
        f = np.eye(8, dtype=float)
        for i in range(4):
            f[i, i + 4] = dt
        return f

    def _predict_in_place(self, dt: float) -> None:
        assert self._x is not None and self._p is not None
        f = self._transition(dt)
        q_scale = max(dt, 1e-3) * self.process_noise
        q = np.diag([q_scale, q_scale, q_scale, q_scale, q_scale * 4, q_scale * 4, q_scale * 4, q_scale * 4])
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

    def _update(self, detection: Detection) -> None:
        assert self._x is not None and self._p is not None
        cx, cy = detection.center
        w, h = detection.size
        z = np.array([cx, cy, w, h], dtype=float)
        h_mat = np.zeros((4, 8), dtype=float)
        h_mat[0, 0] = 1.0
        h_mat[1, 1] = 1.0
        h_mat[2, 2] = 1.0
        h_mat[3, 3] = 1.0
        r = np.eye(4, dtype=float) * self.measurement_noise
        y = z - h_mat @ self._x
        s = h_mat @ self._p @ h_mat.T + r
        k = self._p @ h_mat.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._p = (np.eye(8, dtype=float) - k @ h_mat) @ self._p

    def _predict_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        return self._transition(max(0.0, dt)) @ state

    def _state_to_detection(self, state: np.ndarray, image_width: int, image_height: int) -> Detection:
        cx, cy, w, h = [float(v) for v in state[:4]]
        w = min(max(2.0, w), float(image_width))
        h = min(max(2.0, h), float(image_height))
        x1 = min(max(0.0, cx - w / 2.0), float(image_width - 1))
        y1 = min(max(0.0, cy - h / 2.0), float(image_height - 1))
        x2 = min(max(x1 + 1.0, cx + w / 2.0), float(image_width))
        y2 = min(max(y1 + 1.0, cy + h / 2.0), float(image_height))
        return Detection((x1, y1, x2, y2), self._last_confidence, self._last_class_id)

