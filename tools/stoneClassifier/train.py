"""Train the stone classifier on one extracted dataset version.

Trains on every image set except the test sets, and measures the test sets after every epoch.
Writes into a new run directory:
  model.pt  the final weights, width, class names and test sets; load with model.load_model()
  run.json  dataset hash, code version, parameters, the test metrics of every epoch, and the final confusion matrices
            on the test sets and on the (unaugmented) training sets
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from stone_classifier.learning.dataset import StoneDataset, image_sets
from stone_classifier.learning.metrics import confusion_matrix, format_confusion, format_recall, predict, recall
from stone_classifier.learning.model import StoneNet, pick_device, save_model
from stone_classifier.provenance import git_head, sha256


def main():
    parser = make_parser()
    args = parser.parse_args()

    # 1. Split the image sets: everything not held out for testing is training data.
    all_sets = image_sets(args.dataset)
    if unknown := set(args.test_sets) - set(all_sets):
        parser.error(f"unknown test sets {sorted(unknown)}, available: {all_sets}")
    train_sets = [name for name in all_sets if name not in args.test_sets]
    args.run.mkdir(parents=True)  # Fails if it exists, before spending time on training.

    # 2. Load the data. Only the training data is augmented.
    started = time.monotonic()
    device = pick_device(args.device)
    configure_torch(device, args.threads)
    torch.manual_seed(args.seed)
    train_data = StoneDataset(args.dataset, train_sets, augment=True, colour_jitter=args.colour_jitter)
    test_data = StoneDataset(args.dataset, args.test_sets, augment=False)
    names = train_data.class_names
    counts = torch.bincount(train_data.labels, minlength=len(names))
    if (counts == 0).any():
        parser.error(f"training sets {train_sets} contain no {[n for n, c in zip(names, counts) if c == 0]} patches")

    print(f"train on {train_sets}: {dict(zip(names, counts.tolist()))}")
    print(f"test on {args.test_sets}: {len(test_data)} patches, device {device}")

    # 3. Set up the model, the loss and the optimiser.
    model = StoneNet(len(names), args.width).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(counts).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    # 4. Train, measuring the test sets after every epoch.
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, confusion = measure(model, test_loader, loss_fn, device, len(names))

        test_recall = dict(zip(names, recall(confusion).tolist()))
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_recall": test_recall})
        print(f"epoch {epoch:3d}  train loss {train_loss:.3f}  test loss {test_loss:.3f}  "
              f"recall {format_recall(confusion, names)}")
    print(format_confusion(confusion, names))

    # 5. Measure the training sets without augmentation: errors here mean the model can't even fit what it has seen.
    train_eval_data = StoneDataset(args.dataset, train_sets, augment=False)
    _, train_confusion = measure(model, DataLoader(train_eval_data, batch_size=256), loss_fn, device, len(names))
    print(f"training sets: recall {format_recall(train_confusion, names)}")

    # 6. Save the model and a record of how it was made.
    save_model(args.run / "model.pt", model, names, args.test_sets)
    results = {
        "final_confusion": confusion.tolist(), "final_train_confusion": train_confusion.tolist(),
        "seconds": round(time.monotonic() - started), "history": history,
    }
    record = run_record(args, device, train_sets, dict(zip(names, counts.tolist())), results)
    (args.run / "run.json").write_text(json.dumps(record, indent=2) + "\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="stones.npz of an extracted dataset version")
    parser.add_argument("run", type=Path, help="new output directory for this training run")
    parser.add_argument("--test-sets", nargs="+", default=["angled_hard_lighting"], help="image sets held out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--colour-jitter", type=float, default=0.0, help="random per-channel gain in [1 - j, 1 + j]")
    parser.add_argument("--width", type=int, default=16, help="channels of the first convolution block")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available")
    parser.add_argument("--threads", type=int, help="CPU threads of this run (default: PyTorch's choice)")
    return parser


def configure_torch(device: str, threads: int | None):
    """The thread count and the GPU's algorithm choice both change the order in which numbers are added up, and so
    the result. Fixing them makes a run reproducible."""
    if threads:
        torch.set_num_threads(threads)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Required by deterministic cuBLAS.
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse class frequency, so every class contributes equally to the loss however rare it is."""
    return counts.sum() / (len(counts) * counts.float())


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                device: str) -> float:
    """One pass over the training data, one gradient step per batch. Returns the mean loss."""
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)  # Forward pass: predict and measure the error.
        optimizer.zero_grad()  # Gradients add up by default, so clear the previous batch's.
        loss.backward()  # Backpropagation: d(loss)/d(weight) for every weight.
        optimizer.step()  # Move every weight against its gradient.
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def measure(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str,
            classes: int) -> tuple[float, torch.Tensor]:
    """Mean loss and confusion matrix, without learning anything."""
    logits, labels = predict(model, loader, device)
    loss = loss_fn(logits.to(device), labels.to(device)).item()
    return loss, confusion_matrix(logits.argmax(dim=1), labels, classes)


def run_record(args: argparse.Namespace, device: str, train_sets: list[str], train_counts: dict[str, int],
               results: dict) -> dict:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"path": str(args.dataset.resolve()), "sha256": sha256(args.dataset)},
        "code": {
            "git_head": git_head(Path(__file__).resolve().parent), "torch": torch.__version__,
            "device": device, "threads": torch.get_num_threads(),
        },
        "parameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "colour_jitter": args.colour_jitter, "width": args.width,
        },
        "train_sets": {"sets": train_sets, "counts": train_counts},
        "test_sets": args.test_sets,
        **results,
    }


if __name__ == "__main__":
    main()
