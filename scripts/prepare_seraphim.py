"""Turn the downloaded Seraphim HF snapshot into a plain YOLO dataset tree.

Seraphim ships its 83k images as batched zip archives split by
train/test x images/labels. This script extracts them into the layout ultralytics
expects, verifies that every image has a label, and writes a data.yaml.

    python scripts/prepare_seraphim.py --snapshot datasets/seraphim --out datasets/seraphim_yolo

Seraphim's `test` split becomes our `val` — it is the only held-out data the
dataset provides.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.config import save_json  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify(zip_path: Path) -> tuple[str, str] | None:
    """Map an archive path to (split, kind) using its directory components."""
    parts = [p.lower() for p in zip_path.parts]
    text = "/".join(parts)
    if "label" in text:
        kind = "labels"
    elif "image" in text:
        kind = "images"
    else:
        return None
    if "train" in text:
        split = "train"
    elif "test" in text or "val" in text:
        split = "val"
    else:
        return None
    return split, kind


def extract_all(snapshot: Path, staging: Path) -> tuple[dict[tuple[str, str], int], list[str]]:
    counts: Counter[tuple[str, str]] = Counter()
    skipped: list[str] = []
    archives = sorted(snapshot.rglob("*.zip"))
    if not archives:
        raise SystemExit(f"no .zip archives under {snapshot} — did the download finish?")
    print(f"found {len(archives)} archives")
    for archive in archives:
        rel = archive.relative_to(snapshot)
        target = classify(rel)
        if target is None:
            print(f"  skip (unclassified): {rel}")
            continue
        split, kind = target
        dest = staging / split / kind
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {rel} -> {split}/{kind}")
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    name = Path(member.filename).name
                    if not name or name.startswith("."):
                        continue
                    suffix = Path(name).suffix.lower()
                    if kind == "images" and suffix not in IMAGE_SUFFIXES:
                        continue
                    if kind == "labels" and suffix != ".txt":
                        continue
                    out_path = dest / name
                    with zf.open(member) as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    counts[(split, kind)] += 1
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            # A partially downloaded archive must not abort the whole run — report
            # it and build the dataset from whatever is complete.
            print(f"  SKIP (unreadable, still downloading?): {rel} [{type(exc).__name__}]")
            skipped.append(str(rel))
    return dict(counts), skipped


def pair_and_validate(staging: Path, out_root: Path, single_class: bool) -> dict:
    stats = {
        "train": {"paired": 0, "image_without_label": 0, "label_without_image": 0},
        "val": {"paired": 0, "image_without_label": 0, "label_without_image": 0},
    }
    class_ids: Counter[str] = Counter()
    empty_labels = 0

    for split in ("train", "val"):
        img_dir = staging / split / "images"
        lbl_dir = staging / split / "labels"
        if not img_dir.exists():
            continue
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
        seen: set[str] = set()
        for image in sorted(img_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels.get(image.stem)
            if label is None:
                stats[split]["image_without_label"] += 1
                continue
            seen.add(image.stem)

            text = label.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                empty_labels += 1
            lines = []
            for line in text.splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                class_ids[fields[0]] += 1
                if single_class:
                    fields[0] = "0"
                lines.append(" ".join(fields[:5]))

            shutil.move(str(image), out_img / image.name)
            (out_lbl / f"{image.stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            stats[split]["paired"] += 1

        stats[split]["label_without_image"] = len(set(labels) - seen)

    stats["class_ids_seen"] = dict(class_ids)
    stats["empty_labels"] = empty_labels
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(ROOT / "datasets" / "seraphim"))
    parser.add_argument("--out", default=str(ROOT / "datasets" / "seraphim_yolo"))
    parser.add_argument("--staging", default=None)
    parser.add_argument(
        "--keep-multiclass",
        action="store_true",
        help="preserve original class ids instead of collapsing everything to 0 (drone)",
    )
    parser.add_argument("--clean", action="store_true", help="wipe the output tree first")
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    out_root = Path(args.out)
    staging = Path(args.staging) if args.staging else out_root.parent / f"{out_root.name}_staging"

    if args.clean:
        shutil.rmtree(out_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)

    extracted, skipped = extract_all(snapshot, staging)
    print(f"extracted: { {f'{s}/{k}': v for (s, k), v in extracted.items()} }")
    if skipped:
        print(f"unreadable archives skipped: {skipped}")

    stats = pair_and_validate(staging, out_root, single_class=not args.keep_multiclass)
    shutil.rmtree(staging, ignore_errors=True)

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

    stats["data_yaml"] = str(data_yaml)
    stats["skipped_archives"] = skipped
    save_json(out_root / "prepare_stats.json", stats)
    print(f"train paired: {stats['train']['paired']}")
    print(f"val   paired: {stats['val']['paired']}")
    print(f"class ids seen: {stats['class_ids_seen']}")
    print(f"empty labels (pure background frames): {stats['empty_labels']}")
    for split in ("train", "val"):
        orphan_i = stats[split]["image_without_label"]
        orphan_l = stats[split]["label_without_image"]
        if orphan_i:
            print(f"WARNING {split}: {orphan_i} images have no label file — these were dropped")
        if orphan_l:
            # Expected when only some image batches were downloaded: the labels
            # archive always covers the full 75k/8.3k set.
            print(f"note  {split}: {orphan_l} labels have no image (partial image download — fine)")
    print(f"data_yaml={data_yaml}")


if __name__ == "__main__":
    main()
