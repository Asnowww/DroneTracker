"""Compose a real+sim training set for the stage-2 fine-tune.

Training only on AirSim frames produces a detector that scores 0.99 in the
simulator and collapses on real footage. Training only on real photos leaves the
AirSim regression suite unusable. Stage 2 therefore trains on a mixture, so the
final weights serve both the field deployment and the closed-loop test harness.

Files are hard-linked (same volume, no extra disk) with a copy fallback.

    python scripts/build_mixed_dataset.py --real-samples 18000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs for a split; skips images with no label file."""
    img_dir = dataset / "images" / split
    lbl_dir = dataset / "labels" / split
    pairs = []
    if not img_dir.exists():
        return pairs
    for image in sorted(img_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = lbl_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def emit(pairs: list[tuple[Path, Path]], out_root: Path, split: str, prefix: str) -> int:
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        stem = f"{prefix}_{image.stem}"
        link_or_copy(image, out_img / f"{stem}{image.suffix}")
        link_or_copy(label, out_lbl / f"{stem}.txt")
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default=str(ROOT / "datasets" / "seraphim_yolo"))
    parser.add_argument("--sim", default=str(ROOT / "datasets" / "airsim_drone"))
    parser.add_argument("--out", default=str(ROOT / "datasets" / "mixed"))
    parser.add_argument(
        "--real-samples",
        type=int,
        default=18000,
        help="how many real TRAIN images to mix in (0 = all). Val is never subsampled.",
    )
    parser.add_argument("--sim-samples", type=int, default=0, help="0 = all sim train frames")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    real_root, sim_root, out_root = Path(args.real), Path(args.sim), Path(args.out)
    for path in (real_root, sim_root):
        if not path.exists():
            raise SystemExit(f"missing dataset: {path}")
    if args.clean:
        shutil.rmtree(out_root, ignore_errors=True)

    rnd = random.Random(args.seed)
    stats: dict[str, object] = {}

    real_train = collect(real_root, "train")
    sim_train = collect(sim_root, "train")
    if args.real_samples and args.real_samples < len(real_train):
        real_train = rnd.sample(real_train, args.real_samples)
    if args.sim_samples and args.sim_samples < len(sim_train):
        sim_train = rnd.sample(sim_train, args.sim_samples)

    stats["train_real"] = emit(real_train, out_root, "train", "real")
    stats["train_sim"] = emit(sim_train, out_root, "train", "sim")

    # Validation keeps both domains whole and separable, so the training summary can
    # report real-domain and sim-domain accuracy independently.
    stats["val_real"] = emit(collect(real_root, "val"), out_root, "val", "real")
    stats["val_sim"] = emit(collect(sim_root, "val"), out_root, "val", "sim")

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )

    total_train = stats["train_real"] + stats["train_sim"]
    stats["real_to_sim_ratio"] = round(stats["train_real"] / max(1, stats["train_sim"]), 2)
    stats["train_total"] = total_train
    stats["data_yaml"] = str(data_yaml)
    save_json(out_root / "mix_stats.json", stats)

    print(f"train: {stats['train_real']} real + {stats['train_sim']} sim = {total_train}")
    print(f"val:   {stats['val_real']} real + {stats['val_sim']} sim")
    print(f"real:sim ratio = {stats['real_to_sim_ratio']}:1")
    print(f"data_yaml={data_yaml}")


if __name__ == "__main__":
    main()
