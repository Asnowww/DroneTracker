"""Audit a collected YOLO dataset: verify every positive really shows a drone.

Three escalating levels; only frames flagged by the cheap levels reach the
expensive one:

  1. deterministic  — flat/degenerate image check; labelled crop must contain
                      more edge structure than the ring of background around it.
  2. yolo           — an INDEPENDENT drone detector (e.g. the Hugging Face
                      YOLOv11x weights, trained on real photos, no shared code
                      with our labelling pipeline) must roughly agree with the
                      auto-label. Agreement -> high-confidence pass.
  3. claude sheets  — every frame still unresolved is tiled into contact sheets
                      for visual review by Claude (or a human). The reviewer
                      writes the bad frame stems into a text file, and
                      ``--apply <file>`` quarantines them.

Usage:
    python scripts/audit_dataset.py --dataset datasets/airsim_drone \
        --yolo-weights weights/hf/yolov11x/weight/best.pt
    # review audit/sheets/*.jpg, write bad stems into audit/quarantine_list.txt
    python scripts/audit_dataset.py --dataset datasets/airsim_drone \
        --apply audit/quarantine_list.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402
from drone_tracker.labeling import (  # noqa: E402
    bbox_iou,
    box_structure_scores,
    degenerate_frame_reason,
)

SHEET_COLS, SHEET_ROWS = 5, 5
TILE = 224


def parse_label(path: Path, width: int, height: int) -> tuple[float, float, float, float] | None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.splitlines()[0].split()
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def collect_entries(dataset: Path) -> list[dict]:
    entries = []
    for split in ("train", "val"):
        for image_path in sorted((dataset / "images" / split).glob("*.jpg")):
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            entries.append({"split": split, "image": image_path, "label": label_path})
    return entries


def level1(entries: list[dict]) -> None:
    import cv2

    for entry in entries:
        image = cv2.imread(str(entry["image"]))
        if image is None:
            entry["flags"] = ["unreadable"]
            continue
        height, width = image.shape[:2]
        flags: list[str] = []
        reason = degenerate_frame_reason(image)
        if reason is not None:
            flags.append(f"degenerate:{reason}")
        bbox = parse_label(entry["label"], width, height)
        entry["bbox"] = bbox
        if bbox is not None:
            inside, ring = box_structure_scores(image, bbox)
            entry["edge_inside"], entry["edge_ring"] = round(inside, 2), round(ring, 2)
            # A drone crop has visibly more structure than the sky/ground behind it.
            if inside < 4.0 or inside < ring * 1.05:
                flags.append("low_structure")
        entry["flags"] = flags


def level2(entries: list[dict], weights: str, device: str, conf: float, min_iou: float) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    positives = [e for e in entries if e.get("bbox") is not None and "unreadable" not in e["flags"]]
    batch = 32
    for i in range(0, len(positives), batch):
        chunk = positives[i : i + batch]
        results = model.predict(
            [str(e["image"]) for e in chunk],
            conf=conf,
            imgsz=960,
            device=device,
            verbose=False,
        )
        for entry, result in zip(chunk, results):
            best = 0.0
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                    best = max(best, bbox_iou(tuple(xyxy), entry["bbox"]))
            entry["yolo_iou"] = round(best, 3)
            if best >= min_iou:
                # Independent detector agrees -> clear any structural suspicion.
                entry["flags"] = [f for f in entry["flags"] if f != "low_structure"]
            else:
                entry["flags"].append("yolo_disagrees")


def build_sheets(entries: list[dict], out_dir: Path, sample_rate: float, seed: int = 7) -> list[Path]:
    import random

    import cv2

    rng = random.Random(seed)
    flagged = [e for e in entries if e["flags"]]
    passed = [e for e in entries if not e["flags"] and e.get("bbox") is not None]
    sampled = rng.sample(passed, max(0, min(len(passed), int(len(passed) * sample_rate))))
    review = flagged + sampled
    if not review:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    per_sheet = SHEET_COLS * SHEET_ROWS
    for sheet_idx in range(0, len(review), per_sheet):
        chunk = review[sheet_idx : sheet_idx + per_sheet]
        canvas = np.full((SHEET_ROWS * TILE, SHEET_COLS * TILE, 3), 32, dtype=np.uint8)
        for i, entry in enumerate(chunk):
            row, col = divmod(i, SHEET_COLS)
            image = cv2.imread(str(entry["image"]))
            if image is None:
                continue
            height, width = image.shape[:2]
            bbox = entry.get("bbox")
            if bbox is not None:
                # Crop 3x the box, centred, so the reviewer sees context.
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                half = max(x2 - x1, y2 - y1, 40) * 1.5
                cx1, cy1 = int(max(0, cx - half)), int(max(0, cy - half))
                cx2, cy2 = int(min(width, cx + half)), int(min(height, cy + half))
                crop = image[cy1:cy2, cx1:cx2].copy()
                sx = TILE / max(1, crop.shape[1])
                sy = TILE / max(1, crop.shape[0])
                cv2.rectangle(
                    crop,
                    (int(x1 - cx1), int(y1 - cy1)),
                    (int(x2 - cx1), int(y2 - cy1)),
                    (0, 255, 0),
                    max(1, int(1 / min(sx, sy))),
                )
                tile = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)
            else:
                tile = cv2.resize(image, (TILE, TILE), interpolation=cv2.INTER_AREA)
            label = entry["image"].stem.replace("frame_", "")
            status = ",".join(entry["flags"]) if entry["flags"] else "sample"
            cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(tile, status[:28], (4, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 160, 255), 1)
            canvas[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE] = tile
        sheet_path = out_dir / f"sheet_{sheet_idx // per_sheet:03d}.jpg"
        cv2.imwrite(str(sheet_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(sheet_path)
    return sheets


def apply_quarantine(dataset: Path, list_file: Path) -> None:
    stems = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    quarantine = dataset / "quarantine"
    moved = 0
    for stem in stems:
        for split in ("train", "val"):
            image = dataset / "images" / split / f"{stem}.jpg"
            label = dataset / "labels" / split / f"{stem}.txt"
            if image.exists():
                (quarantine / "images").mkdir(parents=True, exist_ok=True)
                (quarantine / "labels").mkdir(parents=True, exist_ok=True)
                image.rename(quarantine / "images" / image.name)
                if label.exists():
                    label.rename(quarantine / "labels" / label.name)
                moved += 1
    print(f"quarantined {moved} frames -> {quarantine}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--yolo-weights", default=None, help="independent detector for level 2")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-min-iou", type=float, default=0.2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sample-rate", type=float, default=0.05, help="clean frames sampled into sheets")
    parser.add_argument("--apply", default=None, help="quarantine the stems listed in this file and exit")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if args.apply:
        apply_quarantine(dataset, Path(args.apply))
        return

    entries = collect_entries(dataset)
    positives = [e for e in entries if e["label"].read_text(encoding="utf-8").strip()]
    print(f"dataset={dataset} frames_with_labels={len(entries)} positives={len(positives)}")

    level1(entries)
    flagged_l1 = sum(1 for e in entries if e["flags"])
    print(f"level1 flagged: {flagged_l1}")

    if args.yolo_weights:
        level2(entries, args.yolo_weights, args.device, args.yolo_conf, args.yolo_min_iou)
        print(f"level2 flagged: {sum(1 for e in entries if e['flags'])}")
    else:
        print("level2 skipped (no --yolo-weights)")

    audit_dir = dataset / "audit"
    sheets = build_sheets(entries, audit_dir / "sheets", args.sample_rate)

    flagged = [e for e in entries if e["flags"]]
    report = {
        "frames": len(entries),
        "positives": len(positives),
        "flagged": len(flagged),
        "flag_breakdown": {},
        "yolo_iou_mean": None,
        "sheets": [str(s) for s in sheets],
        "flagged_frames": [
            {
                "stem": e["image"].stem,
                "split": e["split"],
                "flags": e["flags"],
                "yolo_iou": e.get("yolo_iou"),
                "edge_inside": e.get("edge_inside"),
                "edge_ring": e.get("edge_ring"),
            }
            for e in flagged
        ],
    }
    for entry in flagged:
        for flag in entry["flags"]:
            report["flag_breakdown"][flag] = report["flag_breakdown"].get(flag, 0) + 1
    ious = [e["yolo_iou"] for e in entries if e.get("yolo_iou") is not None]
    if ious:
        report["yolo_iou_mean"] = round(float(np.mean(ious)), 3)

    save_json(audit_dir / "audit_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "flagged_frames"}, indent=2))
    print(f"report={audit_dir / 'audit_report.json'}")
    if sheets:
        print(f"review sheets: {len(sheets)} images in {audit_dir / 'sheets'}")
        print("next: review the sheets, write bad stems (one per line) into "
              f"{audit_dir / 'quarantine_list.txt'}, then rerun with --apply")


if __name__ == "__main__":
    main()
