"""Train the drone detector on AirSim-collected frames.

Recommended progression (each stage warm-starts from the previous one):

  stage 0  COCO backbone            yolov8s.pt
  stage 1  broad drone prior        Seraphim / HF drone weights   -> generic "drone"
  stage 2  sim domain               AirSim collected frames       -> what flies in AirSim
  stage 3  real domain (field)      real flight footage           -> what flies for real

    python scripts/train_yolo.py --data datasets/airsim_drone/data.yaml --base yolov8s.pt
    python scripts/train_yolo.py --data datasets/airsim_drone/data.yaml \
        --base weights/hf/yolov11x/weight/best.pt --name drone_sim_ft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--base", default="yolov8s.pt", help="base weights to warm-start from")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="drone_airsim")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="dataloader workers. Windows spawns rather than forks, and the "
        "ultralytics default of 8 has been observed to kill workers mid-epoch on "
        "large datasets here; 4 is stable.",
    )
    parser.add_argument("--freeze", type=int, default=0, help="freeze the first N layers")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}. Run scripts/collect_dataset.py first.")

    model = YOLO(args.base)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        freeze=args.freeze or None,
        resume=args.resume,
        # The target is small and far more often distant than close, so keep scale
        # augmentation wide but do not flip vertically — drones are gravity-aligned.
        scale=0.6,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        hsv_v=0.5,
        degrees=10.0,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"

    metrics = YOLO(str(best)).val(data=str(data_path), imgsz=args.imgsz, device=args.device)
    summary = {
        "base": args.base,
        "data": str(data_path),
        "best_weights": str(best),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "map50": float(getattr(metrics.box, "map50", float("nan"))),
        "map50_95": float(getattr(metrics.box, "map", float("nan"))),
        "precision": float(getattr(metrics.box, "mp", float("nan"))),
        "recall": float(getattr(metrics.box, "mr", float("nan"))),
    }
    save_json(save_dir / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"best_weights={best}")


if __name__ == "__main__":
    main()
