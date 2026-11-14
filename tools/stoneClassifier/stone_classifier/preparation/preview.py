"""Renderings for checking data by eye: rectified boards and contact sheets of patches."""

import math
from pathlib import Path

import cv2
import numpy as np

from stone_classifier.preparation.rectify import Grid

LABEL_COLOURS = {"B": (0, 0, 255), "W": (255, 0, 0), ".": (0, 200, 0)}  # BGR
SHEET_COLUMNS, SHEET_ROWS = 20, 20
GAP = 2
CAPTION = 12  # Strip under each tile showing its index into stones.npz.


def draw_board(board: np.ndarray, layout: np.ndarray, grid: Grid) -> np.ndarray:
    """Marks the crop box of every stone and the crop centre of every empty point.
    Boxes for all points would overlap into an unreadable mesh."""
    preview = board.copy()
    h = grid.half_patch
    for (row, col), symbol in np.ndenumerate(layout):
        x, y = grid.centre(row, col)
        if symbol == ".":
            cv2.circle(preview, (x, y), 2, LABEL_COLOURS[symbol], -1)
        else:
            cv2.rectangle(preview, (x - h, y - h), (x + h - 1, y + h - 1), LABEL_COLOURS[symbol], 1)
    return preview


def write_class_sheets(dataset: Path, out: Path):
    """Writes every patch of the dataset into contact sheets `<class>_<page>.png`."""
    data = np.load(dataset)
    patches, labels = data["patches"], data["label"]
    for label, name in enumerate(data["class_names"]):
        write_sheets(patches, np.flatnonzero(labels == label), out, str(name))


def write_sheets(patches: np.ndarray, indices: np.ndarray, out: Path, name: str):
    """Writes the patches at `indices`, in that order, into contact sheets `<name>_<page>.png`."""
    per_page = SHEET_COLUMNS * SHEET_ROWS
    for page, start in enumerate(range(0, len(indices), per_page)):
        sheet = render_sheet(patches, indices[start : start + per_page])
        cv2.imwrite(str(out / f"{name}_{page:03d}.png"), sheet)


def render_sheet(patches: np.ndarray, indices: np.ndarray) -> np.ndarray:
    size = patches.shape[1]
    tile_w, tile_h = size + GAP, size + CAPTION + GAP
    rows, columns = math.ceil(len(indices) / SHEET_COLUMNS), min(len(indices), SHEET_COLUMNS)
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 255, np.uint8)

    for n, index in enumerate(indices):
        y, x = (n // SHEET_COLUMNS) * tile_h, (n % SHEET_COLUMNS) * tile_w
        sheet[y : y + size, x : x + size] = cv2.cvtColor(patches[index], cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, str(index), (x + 1, y + size + CAPTION - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
    return sheet
