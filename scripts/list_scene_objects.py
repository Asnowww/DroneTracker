"""Print AirSim scene object names so you can find the target drone's mesh.

`simSetSegmentationObjectID` matches against these names. If the default
`Target[\\w]*` regex matches nothing, run this and pick the right pattern.

    python scripts/list_scene_objects.py --regex ".*"
    python scripts/list_scene_objects.py --regex ".*(Drone|Flying|Quad|Target).*"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_tracker.airsim_io import connect, list_scene_objects, list_vehicles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regex", default=".*")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--filter", default=None, help="extra client-side substring filter")
    parser.add_argument(
        "--instance",
        action="store_true",
        help="list Cosys-AirSim instance-segmentation meshes instead of scene objects",
    )
    args = parser.parse_args()

    client = connect()
    print(f"vehicles={list_vehicles(client)}")

    if args.instance:
        from drone_tracker.airsim_io import list_instance_segmentation_objects

        objects = list_instance_segmentation_objects(client)
        if args.regex != ".*":
            pattern = re.compile(args.regex)
            objects = [name for name in objects if pattern.search(name)]
    else:
        objects = list_scene_objects(client, args.regex)
    if args.filter:
        pattern = re.compile(args.filter, re.IGNORECASE)
        objects = [name for name in objects if pattern.search(name)]

    print(f"matched={len(objects)} (showing up to {args.limit})")
    for name in objects[: args.limit]:
        print(f"  {name}")

    likely = [n for n in objects if re.search(r"target|drone|flying|quad|multirotor", n, re.IGNORECASE)]
    if likely:
        print("\nlikely target meshes:")
        for name in likely[:20]:
            print(f"  {name}")
        print(
            "\nSet segmentation.target_regex in config/dataset_config.json to a regex "
            "matching exactly one of these, e.g. "
            f'"{re.escape(likely[0])}"'
        )


if __name__ == "__main__":
    main()
