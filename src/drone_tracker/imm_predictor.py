from __future__ import annotations

import math

import numpy as np

from .detector import Detection
from .predictor import PredictionResult

# Shared 12-dimensional state so the three models can be mixed directly:
#   [cx, cy, w, h, vx, vy, vw, vh, ax, ay, aw, ah]
_POS = slice(0, 4)
_VEL = slice(4, 8)
_ACC = slice(8, 12)
_DIM = 12
_MEAS = 4


class IMMPredictor:
    """Interacting Multiple Model predictor over YOLO boxes in the image plane.

    A single constant-velocity filter has to trade smoothness against agility: low
    process noise overshoots on hard turns, high process noise chases detection
    jitter. IMM runs three motion hypotheses in parallel and blends them by their
    posterior likelihood, so it stays smooth in cruise and reacts within a frame or
    two when the target breaks.

    Models:
      ``cv``        constant velocity  — cruise
      ``ca``        constant acceleration — accelerating / diving
      ``maneuver``  constant velocity with large process noise — hard turns

    Drop-in for :class:`~drone_tracker.predictor.ImagePlaneKalmanPredictor`: same
    ``step()`` signature, same :class:`PredictionResult`.
    """

    MODELS = ("cv", "ca", "maneuver")

    def __init__(
        self,
        horizon_s: float = 0.2,
        max_prediction_gap_s: float = 0.6,
        measurement_noise: float = 25.0,
        process_noise: float = 80.0,
        cv_scale: float = 1.0,
        ca_scale: float = 4.0,
        maneuver_scale: float = 25.0,
        transition_stay: float = 0.90,
    ) -> None:
        self.horizon_s = float(horizon_s)
        self.max_prediction_gap_s = float(max_prediction_gap_s)
        self.measurement_noise = float(measurement_noise)
        self.process_noise = float(process_noise)
        self.scales = {
            "cv": float(cv_scale),
            "ca": float(ca_scale),
            "maneuver": float(maneuver_scale),
        }

        n = len(self.MODELS)
        stay = float(transition_stay)
        leave = (1.0 - stay) / (n - 1)
        self.transition = np.full((n, n), leave, dtype=float)
        np.fill_diagonal(self.transition, stay)

        self._mu = np.full(n, 1.0 / n, dtype=float)
        self._x: list[np.ndarray] | None = None
        self._p: list[np.ndarray] | None = None
        self._last_timestamp_s: float | None = None
        self._last_measurement_s: float | None = None
        self._last_confidence = 0.0
        self._last_class_id = 0

        self._h = np.zeros((_MEAS, _DIM), dtype=float)
        self._h[0, 0] = self._h[1, 1] = self._h[2, 2] = self._h[3, 3] = 1.0
        self._r = np.eye(_MEAS, dtype=float) * self.measurement_noise

    # ---------------------------------------------------------------- interface

    @property
    def initialized(self) -> bool:
        return self._x is not None and self._p is not None

    @property
    def mode_probabilities(self) -> dict[str, float]:
        return {name: float(p) for name, p in zip(self.MODELS, self._mu)}

    @property
    def dominant_model(self) -> str:
        return self.MODELS[int(np.argmax(self._mu))]

    def reset(self) -> None:
        self._x = None
        self._p = None
        self._mu = np.full(len(self.MODELS), 1.0 / len(self.MODELS), dtype=float)
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
            self._cycle(detection, dt)
            self._last_timestamp_s = float(timestamp_s)
            if detection is not None:
                self._last_measurement_s = float(timestamp_s)
                self._last_confidence = detection.confidence
                self._last_class_id = detection.class_id

        if not self.initialized:
            return PredictionResult(None, False, None, raw_center, None)

        age_s = float(timestamp_s) - float(self._last_measurement_s or timestamp_s)
        if detection is None and age_s > self.max_prediction_gap_s:
            return PredictionResult(None, False, None, raw_center, age_s)

        future = self._predict_ahead(self.horizon_s)
        predicted = self._state_to_detection(future, image_width, image_height)
        return PredictionResult(
            detection=predicted,
            used_prediction=detection is None or self.horizon_s > 0.0,
            predicted_center=predicted.center,
            raw_center=raw_center,
            prediction_age_s=age_s,
        )

    # ------------------------------------------------------------------ internal

    def _initialize(self, detection: Detection, timestamp_s: float) -> None:
        cx, cy = detection.center
        w, h = detection.size
        state = np.zeros(_DIM, dtype=float)
        state[_POS] = [cx, cy, w, h]
        cov = np.diag([100.0] * 4 + [500.0] * 4 + [1000.0] * 4).astype(float)
        self._x = [state.copy() for _ in self.MODELS]
        self._p = [cov.copy() for _ in self.MODELS]
        self._mu = np.full(len(self.MODELS), 1.0 / len(self.MODELS), dtype=float)
        self._last_timestamp_s = float(timestamp_s)
        self._last_measurement_s = float(timestamp_s)
        self._last_confidence = detection.confidence
        self._last_class_id = detection.class_id

    def _transition_matrix(self, dt: float, model: str) -> np.ndarray:
        f = np.eye(_DIM, dtype=float)
        eye4 = np.eye(4, dtype=float)
        f[_POS, _VEL] = dt * eye4
        if model == "ca":
            f[_POS, _ACC] = 0.5 * dt * dt * eye4
            f[_VEL, _ACC] = dt * eye4
        else:
            # Constant-velocity hypotheses carry no acceleration state.
            f[_ACC, _ACC] = np.zeros((4, 4), dtype=float)
        return f

    def _process_noise(self, dt: float, model: str) -> np.ndarray:
        base = max(dt, 1e-3) * self.process_noise * self.scales[model]
        if model == "ca":
            return np.diag([base * 0.25] * 4 + [base] * 4 + [base * 4.0] * 4)
        return np.diag([base * 0.5] * 4 + [base * 4.0] * 4 + [base * 0.05] * 4)

    def _cycle(self, detection: Detection | None, dt: float) -> None:
        assert self._x is not None and self._p is not None
        n = len(self.MODELS)

        # 1. Mixing --------------------------------------------------------
        c_bar = self.transition.T @ self._mu
        c_bar = np.maximum(c_bar, 1e-12)
        mix = (self.transition * self._mu[:, None]) / c_bar[None, :]

        mixed_x, mixed_p = [], []
        for j in range(n):
            xj = sum(mix[i, j] * self._x[i] for i in range(n))
            pj = np.zeros((_DIM, _DIM), dtype=float)
            for i in range(n):
                diff = (self._x[i] - xj).reshape(-1, 1)
                pj += mix[i, j] * (self._p[i] + diff @ diff.T)
            mixed_x.append(xj)
            mixed_p.append(pj)

        # 2. Model-matched predict + update -------------------------------
        likelihoods = np.ones(n, dtype=float)
        new_x, new_p = [], []
        for j, model in enumerate(self.MODELS):
            f = self._transition_matrix(dt, model)
            x = f @ mixed_x[j]
            p = f @ mixed_p[j] @ f.T + self._process_noise(dt, model)

            if detection is not None:
                cx, cy = detection.center
                bw, bh = detection.size
                z = np.array([cx, cy, bw, bh], dtype=float)
                y = z - self._h @ x
                s = self._h @ p @ self._h.T + self._r
                try:
                    s_inv = np.linalg.inv(s)
                except np.linalg.LinAlgError:  # pragma: no cover - guard only
                    s_inv = np.linalg.pinv(s)
                k = p @ self._h.T @ s_inv
                x = x + k @ y
                p = (np.eye(_DIM, dtype=float) - k @ self._h) @ p
                likelihoods[j] = self._gaussian_likelihood(y, s, s_inv)

            new_x.append(x)
            new_p.append(p)

        self._x, self._p = new_x, new_p

        # 3. Mode probability update --------------------------------------
        if detection is not None:
            posterior = likelihoods * c_bar
            total = float(posterior.sum())
            if total > 1e-300 and math.isfinite(total):
                self._mu = posterior / total
            else:
                self._mu = c_bar / max(float(c_bar.sum()), 1e-12)
        else:
            self._mu = c_bar / max(float(c_bar.sum()), 1e-12)

    @staticmethod
    def _gaussian_likelihood(y: np.ndarray, s: np.ndarray, s_inv: np.ndarray) -> float:
        det = float(np.linalg.det(s))
        if det <= 0.0 or not math.isfinite(det):
            return 1e-12
        exponent = -0.5 * float(y.T @ s_inv @ y)
        exponent = max(exponent, -700.0)
        value = math.exp(exponent) / math.sqrt(((2.0 * math.pi) ** _MEAS) * det)
        return max(value, 1e-12)

    def _combined_state(self) -> np.ndarray:
        assert self._x is not None
        return sum(self._mu[j] * self._x[j] for j in range(len(self.MODELS)))

    def _predict_ahead(self, horizon_s: float) -> np.ndarray:
        """Propagate each model forward, then blend by mode probability.

        Blending the propagated states (rather than propagating the blended state)
        keeps the acceleration hypothesis from leaking into the CV forecast.
        """
        assert self._x is not None
        dt = max(0.0, float(horizon_s))
        result = np.zeros(_DIM, dtype=float)
        for j, model in enumerate(self.MODELS):
            result += self._mu[j] * (self._transition_matrix(dt, model) @ self._x[j])
        return result

    def _state_to_detection(self, state: np.ndarray, image_width: int, image_height: int) -> Detection:
        cx, cy, w, h = (float(v) for v in state[_POS])
        w = min(max(2.0, w), float(image_width))
        h = min(max(2.0, h), float(image_height))
        x1 = min(max(0.0, cx - w / 2.0), float(image_width - 1))
        y1 = min(max(0.0, cy - h / 2.0), float(image_height - 1))
        x2 = min(max(x1 + 1.0, cx + w / 2.0), float(image_width))
        y2 = min(max(y1 + 1.0, cy + h / 2.0), float(image_height))
        return Detection((x1, y1, x2, y2), self._last_confidence, self._last_class_id)
