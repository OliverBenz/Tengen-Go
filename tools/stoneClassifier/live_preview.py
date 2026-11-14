"""Camera preview with the detected stones outlined: black in red, white in blue.

  live_preview.py [--camera 0] [--model stoneNet.onnx]

Click the 4 outermost grid intersections once the board is in view. The grid stays fixed, so keep the camera still.
The buttons below the image pick the board size and start a new corner selection, e.g. after bumping the board.
Keys: r = pick the corners again, q or closing the window = quit.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stone_classifier.learning.dataset import INPUT_SIZE
from stone_classifier.preparation.rectify import Grid, canonical_corners, cut_patch, order_clockwise, rectify

BOARD_SIZES = (9, 13, 19)
STABLE_FRAMES = 5  # A new class must be predicted this many frames in a row, so a glare flash does not show.
SPACING = 32  # Grid spacing of the rectified board in px, as in training.
OUTLINE = {"black": (0, 0, 255), "white": (255, 0, 0)}  # BGR, by class name. Empty points get none.
GRID_COLOUR, CLICK_COLOUR = (255, 0, 255), (0, 255, 255)
WINDOW = "stones"
BAR_HEIGHT, BUTTON_WIDTH = 40, 110  # The button bar below the camera image.
RESELECT = "reselect"
MODEL = Path(__file__).resolve().parents[2] / "src/vision/core/models/stoneNet.onnx"  # Its usage: stoneNet.json.


def make_grid(size: int) -> Grid:
    return Grid(size, SPACING, half_patch=INPUT_SIZE // 2)  # As in training: the network sees exactly one patch.


class Board:
    """Where the intersections of the board are in the camera image, fixed by the 4 clicked grid corners."""

    def __init__(self, corners: np.ndarray, grid: Grid):
        self.corners, self.grid = corners, grid
        to_image = cv2.getPerspectiveTransform(canonical_corners(grid), order_clockwise(corners))
        canonical = np.float32([grid.centre(row, col) for row in range(grid.size) for col in range(grid.size)])
        self.points = cv2.perspectiveTransform(canonical[None], to_image)[0].reshape(grid.size, grid.size, 2)
        # Just under half the local spacing, which shrinks towards the far side of the board.
        self.radii = 0.45 * np.linalg.norm(np.diff(self.points, axis=1), axis=2).mean(axis=1)

    def draw(self, image: np.ndarray, classes: np.ndarray, class_names: list[str]):
        """Thin grid lines, to see at a glance if they fit the board's, and an outline around every stone."""
        lines = np.rint(self.points).astype(np.int32)
        rows = [lines[i] for i in range(self.grid.size)]
        columns = [np.ascontiguousarray(lines[:, i]) for i in range(self.grid.size)]
        cv2.polylines(image, rows + columns, False, GRID_COLOUR, 1)
        for (row, col), label in np.ndenumerate(classes):
            if colour := OUTLINE.get(class_names[label]):
                cv2.circle(image, tuple(lines[row, col]), round(self.radii[row]), colour, 2)


class Debounce:
    """Per intersection, the class shown: it only changes after STABLE_FRAMES identical predictions in a row."""

    def __init__(self, size: int):
        self.shown = np.zeros((size, size), int)
        self.candidate = np.zeros_like(self.shown)
        self.streak = np.zeros_like(self.shown)

    def update(self, predicted: np.ndarray) -> np.ndarray:
        self.streak = np.where(predicted == self.candidate, self.streak + 1, 1)
        self.candidate = predicted
        self.shown = np.where(self.streak >= STABLE_FRAMES, predicted, self.shown)
        return self.shown


@dataclass
class State:
    """What the user chose so far. The board and the debounce follow from it and are rebuilt when it changes."""

    size: int = BOARD_SIZES[0]
    clicks: list[tuple[int, int]] | None = None
    board: Board | None = None
    debounce: Debounce | None = None

    def __post_init__(self):
        self.reset(self.size, keep_corners=False)

    def reset(self, size: int, keep_corners: bool):
        """A new size keeps the clicked corners: they are the same physical intersections."""
        self.size = size
        if not keep_corners:
            self.clicks = []
        self.debounce = Debounce(size)
        self.board = Board(np.float32(self.clicks), make_grid(size)) if len(self.clicks) == 4 else None

    def click(self, point: tuple[int, int]):
        if len(self.clicks) < 4:
            self.clicks.append(point)
            if len(self.clicks) == 4:
                self.reset(self.size, keep_corners=True)


def buttons() -> list[tuple[str, int, int]]:
    """(action, left, right) of every button in the bar, left to right: the board sizes, then reselect."""
    actions = [*BOARD_SIZES, RESELECT]
    return [(action, i * BUTTON_WIDTH, (i + 1) * BUTTON_WIDTH) for i, action in enumerate(actions)]


def draw_bar(width: int, state: State) -> np.ndarray:
    bar = np.full((BAR_HEIGHT, width, 3), 40, np.uint8)
    for action, left, right in buttons():
        active = action == state.size
        cv2.rectangle(bar, (left + 4, 4), (right - 4, BAR_HEIGHT - 5), (90, 90, 90) if active else (65, 65, 65), -1)
        label = f"{action}x{action}" if action in BOARD_SIZES else "Reselect"
        cv2.putText(bar, label, (left + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLICK_COLOUR if active else (220,) * 3, 1)
    return bar


def on_bar_click(state: State, x: int):
    for action, left, right in buttons():
        if left <= x < right:
            if action == RESELECT:
                state.reset(state.size, keep_corners=False)
            else:
                state.reset(action, keep_corners=True)


def classify(model: cv2.dnn.Net, image: np.ndarray, board: Board) -> np.ndarray:
    """The class index of every intersection, as a (size, size) array. Prepared like the training patches."""
    grid = board.grid
    rectified = cv2.cvtColor(rectify(image, board.corners, grid), cv2.COLOR_BGR2RGB)
    patches = np.stack([cut_patch(rectified, grid, row, col) for row in range(grid.size) for col in range(grid.size)])
    model.setInput(patches.transpose(0, 3, 1, 2).astype(np.float32) / 255)  # NCHW in [0, 1], see stoneNet.json.
    return model.forward().argmax(axis=1).reshape(grid.size, grid.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=MODEL, help="ONNX model, with its .json next to it")
    args = parser.parse_args()

    model = cv2.dnn.readNetFromONNX(str(args.model))
    class_names = json.loads(args.model.with_suffix(".json").read_text())["usage"]["output"]["classes"]
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        parser.error(f"cannot open camera {args.camera}")

    state = State()
    image_height = 0  # Clicks below it are on the button bar.

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y >= image_height:
            on_bar_click(state, x)
        else:
            state.click((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        ok, image = camera.read()
        if not ok:
            break
        image_height = image.shape[0]
        if state.board:
            state.board.draw(image, state.debounce.update(classify(model, image, state.board)), class_names)
        else:
            for click in state.clicks:
                cv2.circle(image, click, 5, CLICK_COLOUR, -1)
            prompt = f"{state.size}x{state.size}: click grid corner {len(state.clicks) + 1} of 4"
            cv2.putText(image, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLICK_COLOUR, 2)
        cv2.imshow(WINDOW, np.vstack([image, draw_bar(image.shape[1], state)]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            state.reset(state.size, keep_corners=False)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
