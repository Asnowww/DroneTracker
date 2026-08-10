"""Score a detector separately on the real and simulated domains.

A single mixed-validation number hides the question that actually matters: did
adapting to AirSim cost real-world accuracy? Running each domain's own data.yaml
answers it directly, and also measures inference latency, which decides whether
the model fits on a companion computer.

    python scripts/eval_domains.py --weights runs/detect/mixed_ft_v8s/weights/best.pt
    python scripts/eval_domains.py --weights ... --compare runs/detect/real_prior_v8s/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402

DOMAINS = {
    "real": ROOT / "datasets" / "seraphim_yolo" / "data.yaml",
    "sim": ROOT / "datasets" / "airsim_drone" / "data.yaml",
}


def evaluate(weights: Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    params = sum(p.numel() for p in model.model.parameters())
    result: dict = {
        "weights": str(weights),
        "params_M": round(params / 1e6, 2),
        "imgsz": imgsz,
        "domains": {},
    }

    for name, data_yaml in DOMAINS.items():
        if not data_yaml.exists():
            print(f"skip {name}: {data_yaml} not found")
            continue
        print(f"\n=== {name} domain: {data_yaml} ===")
        metrics = YOLO(str(weights)).val(
            data=str(data_yaml), imgsz=imgsz, device=device, verbose=False
        )
        result["domains"][name] = {
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }

    # Latency on a synthetic batch — what the onboard control loop has to pay.
    import torch

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    for _ in range(5):
        model.predict(source=dummy, device=device, verbose=False)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        model.predict(source=dummy, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    result["latency_ms"] = {
        "mean": round(float(np.mean(times)), 2),
        "p95": round(float(np.percentile(times, 95)), 2),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--compare", default=None, help="a second checkpoint to score alongside")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=str(ROOT / "runs" / "domain_eval.json"))
    args = parser.parse_args()

    report = {"models": []}
    for weights in [args.weights] + ([args.compare] if args.compare else []):
        report["models"].append(evaluate(Path(weights), args.imgsz, args.device))

    save_json(args.out, report)

    print("\n" + "=" * 78)
    header = f"{'model':<34}{'params':>8}{'real mAP50':>12}{'real 50-95':>12}{'sim mAP50':>11}"
    print(header)
    print("-" * 78)
    for m in report["models"]:
        real = m["domains"].get("real", {})
        sim = m["domains"].get("sim", {})
        print(
            f"{Path(m['weights']).parts[-3]:<34}"
            f"{m['params_M']:>7.1f}M"
            f"{real.get('map50', float('nan')):>12.4f}"
            f"{real.get('map50_95', float('nan')):>12.4f}"
            f"{sim.get('map50', float('nan')):>11.4f}"
        )
    print("=" * 78)
    for m in report["models"]:
        print(f"{Path(m['weights']).parts[-3]}: latency {m['latency_ms']['mean']} ms mean")
    print(f"\nreport={args.out}")


if __name__ == "__main__":
    main()
