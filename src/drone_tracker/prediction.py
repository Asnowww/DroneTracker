from __future__ import annotations

from typing import Protocol

from .detector import Detection
from .predictor import ImagePlaneKalmanPredictor, PredictionResult


class Predictor(Protocol):
    def step(
        self,
        detection: Detection | None,
        timestamp_s: float,
        image_width: int,
        image_height: int,
    ) -> PredictionResult: ...

    def reset(self) -> None: ...


def make_predictor(cfg: dict) -> Predictor | None:
    """Build the predictor named by ``prediction.model`` in the tracking config.

    ``kalman`` (default) is the tuned constant-velocity baseline.
    ``imm`` adds constant-acceleration and hard-manoeuvre hypotheses.
    """
    if not cfg.get("enabled", False):
        return None

    model = str(cfg.get("model", "kalman")).lower()
    horizon_s = float(cfg.get("horizon_s", 0.2))
    max_gap_s = float(cfg.get("max_prediction_gap_s", 0.6))
    process_noise = float(cfg.get("process_noise", 80.0))
    measurement_noise = float(cfg.get("measurement_noise", 25.0))

    if model in ("kalman", "cv", "kf"):
        return ImagePlaneKalmanPredictor(
            horizon_s=horizon_s,
            max_prediction_gap_s=max_gap_s,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
    if model == "imm":
        from .imm_predictor import IMMPredictor

        imm_cfg = cfg.get("imm", {})
        return IMMPredictor(
            horizon_s=horizon_s,
            max_prediction_gap_s=max_gap_s,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            cv_scale=float(imm_cfg.get("cv_scale", 1.0)),
            ca_scale=float(imm_cfg.get("ca_scale", 4.0)),
            maneuver_scale=float(imm_cfg.get("maneuver_scale", 25.0)),
            transition_stay=float(imm_cfg.get("transition_stay", 0.90)),
        )
    raise ValueError(f"unknown prediction.model: {model!r} (expected 'kalman' or 'imm')")


def passthrough(detection: Detection | None) -> PredictionResult:
    """Result shape used when prediction is disabled, so logging stays uniform."""
    return PredictionResult(
        detection=detection,
        used_prediction=False,
        predicted_center=detection.center if detection else None,
        raw_center=detection.center if detection else None,
        prediction_age_s=0.0 if detection else None,
    )
