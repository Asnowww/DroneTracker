from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrackingSample:
    t_s: float
    visible: bool
    center_error_x: float
    center_error_y: float
    center_error_px: float
    distance_m: float | None
    vx: float
    vz: float
    yaw_rate_deg_s: float
    confidence: float | None
    prediction_used: bool = False
    predicted_center_x: float | None = None
    predicted_center_y: float | None = None
    raw_center_x: float | None = None
    raw_center_y: float | None = None
    prediction_age_s: float | None = None


def write_tracking_csv(path: Path, samples: list[TrackingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TrackingSample.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def summarize_tracking(samples: list[TrackingSample]) -> dict:
    if not samples:
        return {
            "samples": 0,
            "visible_rate": 0.0,
            "center_error_mean_px": None,
            "center_error_p95_px": None,
            "lost_count": 0,
            "max_lost_duration_s": 0.0,
            "prediction_used_rate": 0.0,
        }
    visible = [sample for sample in samples if sample.visible]
    errors = sorted(sample.center_error_px for sample in visible)
    lost_count = 0
    max_lost_run = 0
    current = 0
    for sample in samples:
        if sample.visible:
            if current:
                lost_count += 1
                max_lost_run = max(max_lost_run, current)
                current = 0
        else:
            current += 1
    if current:
        lost_count += 1
        max_lost_run = max(max_lost_run, current)
    dt = samples[1].t_s - samples[0].t_s if len(samples) > 1 else 0.0
    prediction_used = sum(1 for sample in samples if sample.prediction_used)
    return {
        "samples": len(samples),
        "visible_rate": len(visible) / len(samples),
        "center_error_mean_px": sum(errors) / len(errors) if errors else None,
        "center_error_p95_px": errors[min(len(errors) - 1, int(0.95 * len(errors)))] if errors else None,
        "lost_count": lost_count,
        "max_lost_duration_s": max_lost_run * dt,
        "prediction_used_rate": prediction_used / len(samples),
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

