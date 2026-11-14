"""The labelled photos: images with grid corners in their `.json` and a dotBW layout.

A photo's layout is its own `<name>.txt`, or else the `board.txt` of its folder.
Each folder is an image set, named by its path below the resources root, e.g. `board_1`.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Photo:
    set_name: str
    image: Path
    geometry: Path
    layout: Path

    @property
    def name(self) -> str:
        return f"{self.set_name}/{self.image.name}"


def find_photos(roots: list[Path]) -> tuple[list[Photo], Counter]:
    """All labelled photos below the roots, and per image set the number of photos still missing labels."""
    photos, unlabelled = [], Counter()
    for root in roots:
        for image in sorted(root.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            set_name = image.parent.relative_to(root).as_posix()
            geometry = image.with_suffix(".json")
            layout = image.with_suffix(".txt")
            if not layout.exists():
                layout = image.parent / "board.txt"

            if layout.exists() and has_grid_corners(geometry):
                photos.append(Photo(set_name, image, geometry, layout))
            else:
                unlabelled[set_name] += 1
    return photos, unlabelled


def has_grid_corners(geometry: Path) -> bool:
    """False for a missing file and for one whose corners aren't filled in yet."""
    return geometry.exists() and np.array(json.loads(geometry.read_text())["gridCorners"]).shape == (4, 2)


def load_geometry(path: Path) -> tuple[int, np.ndarray]:
    """Board size and the 4 outermost grid intersections in photo pixels, in no particular order."""
    geometry = json.loads(path.read_text())
    return geometry["boardSize"], np.float32(geometry["gridCorners"])


def load_layout(path: Path, board_size: int) -> np.ndarray:
    """The dotBW layout as a (board_size, board_size) array of '.', 'B' and 'W'."""
    rows = ["".join(line.split()) for line in path.read_text().splitlines()]
    layout = np.array([list(row) for row in rows if row])
    if layout.shape != (board_size, board_size):
        raise ValueError(f"{path}: layout is {layout.shape}, geometry says {board_size}x{board_size}")
    return layout
