from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.detector import Detection
from drone_tracker.predictor import ImagePlaneKalmanPredictor


def main() -> None:
    predictor = ImagePlaneKalmanPredictor(horizon_s=0.2, max_prediction_gap_s=0.6)
    width, height = 1280, 720
    for i in range(8):
        cx = 500.0 + i * 12.0
        det = Detection((cx - 50.0, 300.0, cx + 50.0, 360.0), 0.9, 0)
        result = predictor.step(det, i * 0.1, width, height)
        assert result.detection is not None
    predicted = result.detection.center[0]
    assert predicted > cx, (predicted, cx)

    result = predictor.step(None, 0.85, width, height)
    assert result.detection is not None
    assert result.used_prediction

    result = predictor.step(None, 1.6, width, height)
    assert result.detection is None
    print("predictor_tests_passed=true")


if __name__ == "__main__":
    main()

