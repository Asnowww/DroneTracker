"""Download open-source drone-detection assets from Hugging Face.

Weights (small, fetched by default):
  doguilmak/Drone-Detection-YOLOv11x   MIT       weight/best.pt   mAP50 0.905
  doguilmak/Drone-Detection-YOLOv8x    unstated  weight/best.pt

Dataset (large, opt-in):
  lgrzybowski/seraphim-drone-detection-dataset   CC BY 4.0
  83,483 images (75,134 train / 8,349 test), single `drone` class, YOLO format, 640x640.

    python scripts/fetch_hf_assets.py --weights
    python scripts/fetch_hf_assets.py --dataset --dataset-dir E:/datasets/seraphim
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402

WEIGHT_REPOS = {
    "yolov11x": ("doguilmak/Drone-Detection-YOLOv11x", "weight/best.pt", "mit"),
    "yolov8x": ("doguilmak/Drone-Detection-YOLOv8x", "weight/best.pt", "unstated"),
}
DATASET_REPO = "lgrzybowski/seraphim-drone-detection-dataset"


def fetch_weights(out_dir: Path, which: list[str]) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for key in which:
        repo_id, filename, license_name = WEIGHT_REPOS[key]
        print(f"downloading {repo_id}/{filename} (license: {license_name}) ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(out_dir / key))
        resolved[key] = path
        print(f"  -> {path}")
        if license_name == "unstated":
            print(
                f"  NOTE: {repo_id} does not declare a license on the Hub. "
                "Confirm terms before using it in anything you ship."
            )
    return resolved


def fetch_dataset(out_dir: Path, patterns: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    print(f"downloading dataset {DATASET_REPO} (CC BY 4.0) -> {out_dir}")
    if patterns:
        print(f"  restricted to patterns: {patterns}")
    else:
        print("  full snapshot is ~9.1 GB — make sure the destination has room.")
    path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=patterns or None,
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    print(f"  -> {path}")
    print(
        "Reminder: this set mixes real photos with marketing and synthetic renders. "
        "Use it for pre-training, then fine-tune on AirSim frames for the sim domain "
        "and on real flight footage before field trials."
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="download pretrained drone detectors")
    parser.add_argument("--which", nargs="*", default=["yolov11x"], choices=sorted(WEIGHT_REPOS))
    parser.add_argument("--weights-dir", default=str(ROOT / "weights" / "hf"))
    parser.add_argument("--dataset", action="store_true", help="download the Seraphim dataset (large)")
    parser.add_argument("--dataset-dir", default=str(ROOT / "datasets" / "seraphim"))
    parser.add_argument(
        "--dataset-patterns",
        nargs="*",
        default=None,
        help="only fetch matching paths, e.g. 'train/images/batch_001.zip' 'train/labels/*' "
        "'test/*'. Two of the four train image batches (~37k images) are already plenty "
        "for a generic drone prior and halve the download.",
    )
    args = parser.parse_args()

    if not args.weights and not args.dataset:
        parser.error("pass --weights and/or --dataset")

    manifest: dict[str, object] = {}
    try:
        if args.weights:
            manifest["weights"] = fetch_weights(Path(args.weights_dir), args.which)
        if args.dataset:
            manifest["dataset"] = fetch_dataset(Path(args.dataset_dir), args.dataset_patterns)
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc

    save_json(ROOT / "weights" / "hf_manifest.json", manifest)
    print(f"manifest={ROOT / 'weights' / 'hf_manifest.json'}")


if __name__ == "__main__":
    main()
