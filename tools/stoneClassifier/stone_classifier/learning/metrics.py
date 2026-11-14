"""Measuring a classifier's predictions."""

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and true labels of every sample in loader order, on the CPU."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def confusion_matrix(predicted: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    """Rows: true class, columns: predicted class."""
    pairs = labels * classes + predicted  # One number per (true, predicted) combination.
    return torch.bincount(pairs, minlength=classes * classes).reshape(classes, classes)


def recall(confusion: torch.Tensor) -> torch.Tensor:
    """Per class: the fraction of its samples classified correctly."""
    return confusion.diag() / confusion.sum(dim=1)


def format_recall(confusion: torch.Tensor, names: list[str]) -> str:
    return "  ".join(f"{name} {value:.3f}" for name, value in zip(names, recall(confusion).tolist()))


def format_confusion(confusion: torch.Tensor, names: list[str]) -> str:
    lines = ["true \\ predicted " + "".join(f"{name:>8}" for name in names)]
    lines += [f"{name:<17}" + "".join(f"{count:>8}" for count in row) for name, row in zip(names, confusion.tolist())]
    return "\n".join(lines)
