"""Offline A/B of the image-plane predictors — runs without AirSim or a GPU.

Simulates a ground-truth box trajectory, corrupts it with detection noise and
dropouts, and scores each predictor on how close its ``horizon_s`` forecast lands
to where the target actually will be. That is exactly the quantity the visual
servo controller consumes, so it is the honest metric for "does prediction help".
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.detector import Detection  # noqa: E402
from drone_tracker.imm_predictor import IMMPredictor  # noqa: E402
from drone_tracker.predictor import ImagePlaneKalmanPredictor  # noqa: E402

WIDTH, HEIGHT = 1280, 720
HORIZON_S = 0.2
DT = 0.05  # 20 Hz control loop
DURATION_S = 24.0
NOISE_PX = 4.0
DROPOUT_RATE = 0.08


def smooth_truth(t: float) -> tuple[float, float, float, float]:
    """Gentle cruise — a constant-velocity model should already be near optimal."""
    cx = WIDTH / 2 + 220.0 * math.sin(0.45 * t)
    cy = HEIGHT / 2 + 70.0 * math.sin(0.31 * t + 0.4)
    w = 130.0 + 25.0 * math.sin(0.2 * t)
    return cx, cy, w, 0.62 * w


def maneuvering_truth(t: float) -> tuple[float, float, float, float]:
    """Hard S-turns with near-square-wave reversals — the constant-velocity killer."""
    cx = WIDTH / 2 + 250.0 * math.tanh(2.6 * math.sin(0.62 * t))
    cy = HEIGHT / 2 + 110.0 * math.tanh(2.2 * math.sin(0.94 * t + 0.7))
    w = 130.0 + 45.0 * math.sin(0.5 * t)
    return cx, cy, w, 0.62 * w


def _detection_from(cx: float, cy: float, w: float, h: float) -> Detection:
    return Detection((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.9, 0)


def run_case(truth_fn, predictor, seed: int) -> tuple[float, float, int]:
    """Return (rmse_px, p95_px, scored_steps) of the horizon forecast."""
    rnd = random.Random(seed)
    errors: list[float] = []
    steps = int(DURATION_S / DT)

    for i in range(steps):
        t = i * DT
        cx, cy, w, h = truth_fn(t)

        if rnd.random() < DROPOUT_RATE:
            detection = None
        else:
            detection = _detection_from(
                cx + rnd.gauss(0.0, NOISE_PX),
                cy + rnd.gauss(0.0, NOISE_PX),
                max(8.0, w + rnd.gauss(0.0, NOISE_PX)),
                max(8.0, h + rnd.gauss(0.0, NOISE_PX)),
            )

        if predictor is None:
            # Baseline: feed the raw detection straight through, which is what the
            # controller sees when prediction is disabled. Hold the last box on a miss.
            result_center = _hold_last(detection, run_case)
        else:
            result = predictor.step(detection, t, WIDTH, HEIGHT)
            result_center = result.detection.center if result.detection is not None else None

        if result_center is None:
            continue
        if t < 2.0:  # let every filter converge before scoring
            continue

        true_cx, true_cy, _, _ = truth_fn(t + HORIZON_S)
        errors.append(math.hypot(result_center[0] - true_cx, result_center[1] - true_cy))

    if not errors:
        return float("inf"), float("inf"), 0
    errors_sorted = sorted(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    p95 = errors_sorted[min(len(errors_sorted) - 1, int(0.95 * len(errors_sorted)))]
    return rmse, p95, len(errors)


def _hold_last(detection: Detection | None, marker) -> tuple[float, float] | None:
    if detection is not None:
        marker._last = detection.center  # type: ignore[attr-defined]
    return getattr(marker, "_last", None)


def make_kalman() -> ImagePlaneKalmanPredictor:
    return ImagePlaneKalmanPredictor(
        horizon_s=HORIZON_S, max_prediction_gap_s=0.6, process_noise=80.0, measurement_noise=25.0
    )


def make_imm() -> IMMPredictor:
    return IMMPredictor(
        horizon_s=HORIZON_S, max_prediction_gap_s=0.6, process_noise=80.0, measurement_noise=25.0
    )


def main() -> None:
    cases = [("smooth", smooth_truth), ("maneuvering", maneuvering_truth)]
    variants = [("none", lambda: None), ("kalman", make_kalman), ("imm", make_imm)]

    results: dict[tuple[str, str], tuple[float, float, int]] = {}
    print(f"{'trajectory':<14}{'predictor':<10}{'rmse_px':>10}{'p95_px':>10}{'steps':>8}")
    print("-" * 52)
    for case_name, truth_fn in cases:
        for variant_name, factory in variants:
            if hasattr(run_case, "_last"):
                del run_case._last  # reset the passthrough baseline between runs
            rmse, p95, steps = run_case(truth_fn, factory(), seed=1234)
            results[(case_name, variant_name)] = (rmse, p95, steps)
            print(f"{case_name:<14}{variant_name:<10}{rmse:>10.2f}{p95:>10.2f}{steps:>8}")

    imm = IMMPredictor(horizon_s=HORIZON_S, max_prediction_gap_s=0.6)
    for i in range(int(8.0 / DT)):
        t = i * DT
        cx, cy, w, h = maneuvering_truth(t)
        imm.step(_detection_from(cx, cy, w, h), t, WIDTH, HEIGHT)
    print(f"\nimm_mode_probabilities_after_maneuver={imm.mode_probabilities}")
    print(f"imm_dominant_model={imm.dominant_model}")

    # Prediction must beat no-prediction on both trajectories: that is the whole point.
    for case_name, _ in cases:
        none_rmse = results[(case_name, "none")][0]
        for variant in ("kalman", "imm"):
            assert results[(case_name, variant)][0] < none_rmse, (
                case_name,
                variant,
                results[(case_name, variant)][0],
                none_rmse,
            )

    # IMM exists to handle manoeuvres; it must not lose to plain CV there.
    assert results[("maneuvering", "imm")][0] <= results[("maneuvering", "kalman")][0], results

    # And it must not pay a large penalty in smooth cruise.
    smooth_imm = results[("smooth", "imm")][0]
    smooth_kf = results[("smooth", "kalman")][0]
    assert smooth_imm <= smooth_kf * 1.35, (smooth_imm, smooth_kf)

    print("\nprediction_benchmark_passed=true")


if __name__ == "__main__":
    main()
