"""Extract one labelled patch per board intersection from the labelled photos, see stone_classifier/preparation/corpus.py.

Each photo is warped onto a canonical grid first, so all patches share one scale.
Writes one frozen dataset version into a new directory:
  stones.npz     patches (N, P, P, 3) uint8 RGB, label, set, image, row, col, class_names
  manifest.json  parameters, class counts, and the hash of every input file and of this script
  check/         rectified boards with their crops marked, and per-class contact sheets of all patches
"""

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from stone_classifier.preparation.preview import draw_board, write_class_sheets
from stone_classifier.preparation.corpus import Photo, find_photos, load_geometry, load_layout
from stone_classifier.provenance import git_head, sha256
from stone_classifier.preparation.rectify import OVERSAMPLE, Grid, cut_patch, orient_layout, rectify

CLASS_NAMES = ["empty", "black", "white"]
LABEL_OF = {".": 0, "B": 1, "W": 2}
COLUMNS = ["patches", "label", "set", "image", "row", "col"]


def main():
    args = parse_args()
    half_patch = round(args.window * args.spacing / 2)

    # 1. Find the labelled photos.
    photos, unlabelled = find_photos(args.resources)
    args.out.mkdir(parents=True)  # Fails if it exists: a written dataset version is frozen.
    (args.out / "check" / "boards").mkdir(parents=True)

    # 2. Per photo: warp the board flat, turn its layout to match, cut one patch per intersection.
    columns = {name: [] for name in COLUMNS}
    sources = []
    for photo in photos:
        board_size, grid_corners = load_geometry(photo.geometry)
        grid = Grid(board_size, args.spacing, half_patch)
        board = rectify(cv2.imread(str(photo.image)), grid_corners, grid)
        layout, rotation, scores = orient_layout(load_layout(photo.layout, board_size), board, grid)
        print(f"{photo.name}: rotation {rotation}, scores {[round(s) for s in scores]}")

        add_patches(columns, photo, cv2.cvtColor(board, cv2.COLOR_BGR2RGB), layout, grid)
        sources.append(source_record(photo, rotation))
        board_preview = args.out / "check" / "boards" / f"{photo.name.replace('/', '-')}.png"
        cv2.imwrite(str(board_preview), draw_board(board, layout, grid))

    # 3. Save the dataset and how it was made.
    dataset = args.out / "stones.npz"
    np.savez_compressed(dataset, **{name: np.array(values) for name, values in columns.items()},
                        class_names=np.array(CLASS_NAMES))
    manifest = make_manifest(args, half_patch, columns, sources)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # 4. Render the patch sheets from the saved file, so they show exactly what training reads.
    write_class_sheets(dataset, args.out / "check")

    for set_name, count in sorted(unlabelled.items()):
        print(f"skipped {set_name}: {count} images without layout or grid corners")
    totals = Counter(columns["label"])
    per_class = {name: totals[label] for label, name in enumerate(CLASS_NAMES)}
    print(f"wrote {len(columns['label'])} patches of {2 * half_patch}x{2 * half_patch} px: {per_class}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_resources = [script_dir.parents[1] / "tests/vision/resources", script_dir / "training_data"]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="new output directory, e.g. .../stones_v1")
    parser.add_argument("--resources", type=Path, nargs="+", default=default_resources, help="labelled photo folders")
    parser.add_argument("--spacing", type=int, default=32, help="grid spacing of the rectified board in px")
    parser.add_argument("--window", type=float, default=1.5, help="patch side in grid spacings")
    return parser.parse_args()


def add_patches(columns: dict[str, list], photo: Photo, board: np.ndarray, layout: np.ndarray, grid: Grid):
    for (row, col), symbol in np.ndenumerate(layout):
        columns["patches"].append(cut_patch(board, grid, row, col))
        columns["label"].append(LABEL_OF[symbol])
        columns["set"].append(photo.set_name)
        columns["image"].append(photo.name)
        columns["row"].append(row)
        columns["col"].append(col)


def source_record(photo: Photo, rotation: int) -> dict:
    files = [photo.image, photo.geometry, photo.layout]
    return {"image": photo.name, "rotation": rotation, "sha256": {path.name: sha256(path) for path in files}}


def make_manifest(args: argparse.Namespace, half_patch: int, columns: dict[str, list], sources: list[dict]) -> dict:
    script = Path(__file__).resolve()
    counts = Counter(zip(columns["set"], columns["label"]))
    sets = sorted(set(columns["set"]))
    return {
        "created": date.today().isoformat(),
        "script": {"sha256": sha256(script), "git_head": git_head(script.parent)},
        "resources": [str(root) for root in args.resources],
        "parameters": {
            "spacing": args.spacing, "window": args.window, "patch_size": 2 * half_patch,
            "oversample": OVERSAMPLE, "channels": "RGB",
        },
        "classes": CLASS_NAMES,
        "counts": {s: {name: counts[s, label] for label, name in enumerate(CLASS_NAMES)} for s in sets},
        "sources": sources,
    }


if __name__ == "__main__":
    main()
