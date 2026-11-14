"""The network input: patches of one extracted dataset version, see extract_patches.py."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_SIZE = 32  # Network input in px. Stored patches are larger, so the crop can move around the intersection.


def image_sets(path: Path) -> list[str]:
    return sorted(set(np.load(path)["set"].tolist()))


class StoneDataset(Dataset):
    """The patches of the given image sets, prepared as network input.

    augment        a new random crop and rotation on every access, instead of the fixed centre crop
    colour_jitter  when augmenting: scale each colour channel by a random factor in [1 - j, 1 + j]
    """

    def __init__(self, path: Path, sets: list[str], augment: bool, colour_jitter: float = 0.0):
        data = np.load(path)
        keep = np.isin(data["set"], sets)

        self.indices = np.flatnonzero(keep)  # Position of each sample in stones.npz, to look it up again.
        self.patches = torch.from_numpy(data["patches"][keep]).permute(0, 3, 1, 2)  # (N, 3, P, P) uint8, channels first for PyTorch
        self.labels = torch.from_numpy(data["label"][keep])
        self.class_names = [str(name) for name in data["class_names"]]
        self.augment, self.colour_jitter = augment, colour_jitter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        patch = self.crop_patch(self.patches[i]).float() / 255
        if self.augment and self.colour_jitter > 0:
            gain = 1 + self.colour_jitter * (2 * torch.rand(3, 1, 1) - 1)
            patch = (patch * gain).clamp(0, 1)
        return patch, self.labels[i]

    def crop_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """The centre crop, or when augmenting a randomly placed crop in a random rotation."""
        margin = patch.shape[-1] - INPUT_SIZE
        if not self.augment:
            c = margin // 2
            return patch[:, c : c + INPUT_SIZE, c : c + INPUT_SIZE]

        y, x = torch.randint(0, margin + 1, (2,)).tolist()  # Moves the intersection off-centre, like grid error does.
        quarter_turns = int(torch.randint(4, ()))  # Stones look the same from every side.
        return torch.rot90(patch[:, y : y + INPUT_SIZE, x : x + INPUT_SIZE], quarter_turns, dims=(1, 2))

