"""The stone classifier: a small CNN mapping an RGB patch to one score (logit) per class."""

from pathlib import Path

import torch
from torch import nn


class StoneNet(nn.Module):
    """Three convolution blocks, each halving the resolution and doubling the channels, then one linear layer on the
    averaged features. `width` is the channel count of the first block.
    Shapes below are for a 32x32 input and width 16; the average pooling makes any input size work."""

    def __init__(self, classes: int = 3, width: int = 16):
        super().__init__()
        self.width = width
        self.features = nn.Sequential(
            block(3, width),              # (N, 3, 32, 32)  -> (N, 16, 16, 16)
            block(width, 2 * width),      # (N, 16, 16, 16) -> (N, 32, 8, 8)
            block(2 * width, 4 * width),  # (N, 32, 8, 8)   -> (N, 64, 4, 4)
        )
        self.classify = nn.Linear(4 * width, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # Global average pool: (N, 64), one value per learned feature.
        return self.classify(x)  # (N, classes) logits; the softmax is part of the loss.


def block(inputs: int, outputs: int) -> nn.Sequential:
    """3x3 convolution, ReLU, then halve the resolution."""
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


def pick_device(requested: str | None = None) -> str:
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def save_model(path: Path, model: StoneNet, class_names: list[str], test_sets: list[str]):
    checkpoint = dict(state_dict=model.state_dict(), width=model.width, class_names=class_names, test_sets=test_sets)
    torch.save(checkpoint, path)


def load_model(path: Path) -> tuple[StoneNet, dict]:
    """The trained model on the CPU, and the checkpoint's other entries: class_names, test_sets."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = StoneNet(len(checkpoint["class_names"]), checkpoint.pop("width"))
    model.load_state_dict(checkpoint.pop("state_dict"))
    return model, checkpoint
