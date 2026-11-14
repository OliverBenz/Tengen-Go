"""Warping a photo onto a canonical grid, so every board has the same intersection spacing in pixels."""

from dataclasses import dataclass

import cv2
import numpy as np

OVERSAMPLE = 4  # Warp at 4x and box-filter down, so thin grid lines don't alias away.


@dataclass(frozen=True)
class Grid:
    """The canonical grid: intersection (row, col) sits at pixel (x, y) = centre(row, col).

    half_patch is also the margin around the outermost intersections, so edge patches fit into the image.
    """

    size: int
    spacing: int
    half_patch: int

    @property
    def side(self) -> int:
        return (self.size - 1) * self.spacing + 2 * self.half_patch

    def centre(self, row: int, col: int) -> tuple[int, int]:
        return self.half_patch + col * self.spacing, self.half_patch + row * self.spacing


def canonical_corners(grid: Grid) -> np.ndarray:
    """The 4 outermost intersections of the canonical grid, clockwise from the top left."""
    first, last = grid.half_patch, grid.half_patch + (grid.size - 1) * grid.spacing
    return np.float32([[first, first], [last, first], [last, last], [first, last]])


def rectify(image: np.ndarray, grid_corners: np.ndarray, grid: Grid) -> np.ndarray:
    """Warps the photo so its grid corners land on the canonical grid's corners."""
    H = cv2.getPerspectiveTransform(order_clockwise(grid_corners), canonical_corners(grid) * OVERSAMPLE)

    side = grid.side * OVERSAMPLE
    fine = cv2.warpPerspective(image, H, (side, side), borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(fine, (grid.side, grid.side), interpolation=cv2.INTER_AREA)


def order_clockwise(corners: np.ndarray) -> np.ndarray:
    """Sorts the corners clockwise on screen. The start corner stays arbitrary, see orient_layout()."""
    centre = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    return corners[np.argsort(angles)]


def orient_layout(layout: np.ndarray, board: np.ndarray, grid: Grid) -> tuple[np.ndarray, int, list[float]]:
    """Turns the layout to match the rectified board, since the labels don't say from which side it was photographed.

    Scores each of the 4 rotations by how well its black stones fall on dark and its white stones on bright
    intersections. Returns the best rotated layout, its number of quarter turns, and all 4 scores.
    """
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = grid.spacing // 4
    centres = [[grid.centre(row, col) for col in range(grid.size)] for row in range(grid.size)]
    brightness = np.array([[gray[y - r : y + r, x - r : x + r].mean() for x, y in row] for row in centres])
    contrast = brightness - np.median(brightness)

    expected = np.select([layout == "B", layout == "W"], [-1.0, 1.0], 0.0)
    scores = [float(np.sum(np.rot90(expected, k) * contrast)) for k in range(4)]
    best = int(np.argmax(scores))
    return np.rot90(layout, best), best, scores


def cut_patch(board: np.ndarray, grid: Grid, row: int, col: int) -> np.ndarray:
    x, y = grid.centre(row, col)
    h = grid.half_patch
    return board[y - h : y + h, x - h : x + h]
