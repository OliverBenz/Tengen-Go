"""Evaluate a trained stone classifier and write contact sheets of the patches it gets wrong.

Writes into a new directory:
  confusion.txt                     confusion matrix, per-class recall, and the photos with mistakes
  <true>_as_<predicted>_<page>.png  misclassified patches, most confident mistakes first,
                                    captioned with their index into stones.npz
"""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stone_classifier.preparation.preview import write_sheets
from stone_classifier.learning.dataset import StoneDataset, image_sets
from stone_classifier.learning.metrics import confusion_matrix, format_confusion, format_recall, predict
from stone_classifier.learning.model import load_model, pick_device


def main():
    parser = make_parser()
    args = parser.parse_args()

    # 1. Load the model, and prepare the patches.
    model, info = load_model(args.model)
    names = info["class_names"]
    sets = args.sets or info["test_sets"]
    if unknown := set(sets) - set(image_sets(args.dataset)):
        parser.error(f"{args.dataset} has no image sets {sorted(unknown)}")
    data = StoneDataset(args.dataset, sets, augment=False)
    if data.class_names != names:
        parser.error(f"model classes {names} differ from dataset classes {data.class_names}")
    args.out.mkdir(parents=True)

    # 2. Predict every patch.
    device = pick_device()
    logits, labels = predict(model.to(device), DataLoader(data, batch_size=256), device)
    confidence, predicted = logits.softmax(dim=1).max(dim=1)

    # 3. Report the confusion matrix, and which photos are not classified perfectly.
    stored = np.load(args.dataset)
    photos = stored["image"][data.indices]
    confusion = confusion_matrix(predicted, labels, len(names))
    report = (f"evaluated on {sets}: {len(data)} patches\n{format_confusion(confusion, names)}\n"
              f"recall {format_recall(confusion, names)}\n{format_photo_mistakes(photos, predicted != labels)}")
    print(report, end="")
    (args.out / "confusion.txt").write_text(report)

    # 4. Contact sheets per kind of mistake, most confident first.
    patches = stored["patches"]
    for true, guess in product(range(len(names)), repeat=2):
        mistakes = torch.nonzero((labels == true) & (predicted == guess)).flatten()
        if true == guess or len(mistakes) == 0:
            continue
        mistakes = mistakes[confidence[mistakes].argsort(descending=True)]
        write_sheets(patches, data.indices[mistakes.numpy()], args.out, f"{names[true]}_as_{names[guess]}")


def format_photo_mistakes(photos: np.ndarray, wrong: torch.Tensor) -> str:
    mistakes = Counter(photos[wrong.numpy()].tolist())
    lines = [f"photos with mistakes: {len(mistakes)} of {len(set(photos.tolist()))}"]
    lines += [f"  {photo}: {count}" for photo, count in sorted(mistakes.items())]
    return "\n".join(lines) + "\n"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="model.pt of a training run")
    parser.add_argument("dataset", type=Path, help="stones.npz to evaluate on")
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--sets", nargs="+", help="image sets to evaluate on (default: the run's test sets)")
    return parser


if __name__ == "__main__":
    main()
