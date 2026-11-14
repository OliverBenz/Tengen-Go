"""Export a trained stone classifier as ONNX, with a JSON describing how to use it and where it came from.

  export_onnx.py <run>/model.pt [--out src/vision/core/models]

Writes <out>/stoneNet.onnx and <out>/stoneNet.json. The run's run.json and its dataset must still exist: the JSON
records the dataset version, and the export is checked against PyTorch on the run's test patches.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

from stone_classifier.learning.dataset import INPUT_SIZE, StoneDataset
from stone_classifier.learning.metrics import predict
from stone_classifier.learning.model import load_model
from stone_classifier.preparation.rectify import OVERSAMPLE
from stone_classifier.provenance import sha256

NAME = "stoneNet"
REPO = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-4  # Largest allowed logit difference between ONNX and PyTorch.


def main():
    args = make_parser().parse_args()

    # 1. Load the model and the record of its training run.
    model, info = load_model(args.model)
    model.eval()
    run = json.loads((args.model.parent / "run.json").read_text())
    dataset = Path(run["dataset"]["path"])
    if sha256(dataset) != run["dataset"]["sha256"]:
        raise SystemExit(f"{dataset} changed since the run, so the export cannot be checked against it")

    # 2. Export, with a free batch size so all intersections of a board run in one call.
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / f"{NAME}.onnx"
    example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(model, (example,), onnx_path, input_names=["patches"], output_names=["logits"],
                      dynamic_shapes={"x": {0: torch.export.Dim("N")}}, external_data=False)
    strip_metadata(onnx_path)

    # 3. Check it the way C++ will run it, through OpenCV, against PyTorch on the run's test patches.
    patches = StoneDataset(dataset, info["test_sets"], augment=False)
    expected, labels = predict(model, DataLoader(patches, batch_size=256), "cpu")
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(torch.stack([patches[i][0] for i in range(len(patches))]).numpy())
    actual = net.forward()
    difference = float(np.abs(actual - expected.numpy()).max())
    errors = int((actual.argmax(axis=1) != labels.numpy()).sum())
    if difference > TOLERANCE:
        onnx_path.unlink()
        raise SystemExit(f"ONNX differs from PyTorch by up to {difference}, nothing written")

    # 4. Describe how to use it and where it came from.
    description = describe(args.model, run, info, errors, len(patches))
    (args.out / f"{NAME}.json").write_text(json.dumps(description, indent=2) + "\n")
    print(f"wrote {onnx_path} and {NAME}.json: {errors} errors on {len(patches)} test patches, "
          f"largest logit difference to PyTorch {difference:.1e}")


def strip_metadata(path: Path):
    """The exporter stores a Python stack trace per node, full of this machine's paths. Only the graph is needed.
    Assumes a plain feed-forward graph: no subgraphs (If, Loop) whose nodes would keep theirs."""
    model = onnx.load(path)
    graph = model.graph
    for item in [model, graph, *graph.node, *graph.input, *graph.output, *graph.value_info, *model.functions,
                 *(node for function in model.functions for node in function.node)]:
        if hasattr(item, "metadata_props"):
            del item.metadata_props[:]
        item.doc_string = ""
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="model.pt of a training run, next to its run.json")
    parser.add_argument("--out", type=Path, default=REPO / "src/vision/core/models")
    return parser


def describe(model: Path, run: dict, info: dict, errors: int, patches: int) -> dict:
    dataset = Path(run["dataset"]["path"])
    parameters = run["parameters"]
    return {
        "model": NAME,
        "dataset_version": dataset.parent.name,
        "usage": {
            "summary": "Classifies one board intersection of a photo as empty, black or white.",
            "geometry": f"Warp the photo onto a square grid with neighbouring intersections {run_spacing(dataset)} px "
                        f"apart. Warp at {OVERSAMPLE}x that size, then shrink with area interpolation, so thin grid "
                        "lines don't alias away.",
            "patch": f"{INPUT_SIZE}x{INPUT_SIZE} px of the warped image, centred on the intersection.",
            "input": {"name": "patches", "shape": ["N", 3, INPUT_SIZE, INPUT_SIZE], "layout": "NCHW",
                      "colour": "RGB", "scale": "8-bit value / 255, so in [0, 1]", "normalisation": "none"},
            "output": {"name": "logits", "shape": ["N", len(info["class_names"])], "classes": info["class_names"],
                       "decision": "argmax; softmax gives a confidence"},
            "rotation": "Any: stones look the same from every side, so the board may be seen from any side.",
            "video": "Debounce each intersection over a few frames: glare can show a false white for 1-2 frames.",
        },
        "training": {
            "run": f"{model.parent.parent.name}/{model.parent.name}",
            "dataset_sha256": run["dataset"]["sha256"],
            "augmentation": {"random_crop": True, "quarter_turns": True,
                             "colour_jitter": parameters.get("colour_jitter", 0.0)},
            "width": parameters["width"],
            "epochs": parameters["epochs"],
            "seed": parameters["seed"],
            "test_sets": info["test_sets"],
            "test_errors": f"{errors} of {patches} patches",
        },
    }


def run_spacing(dataset: Path) -> int:
    return json.loads((dataset.parent / "manifest.json").read_text())["parameters"]["spacing"]


if __name__ == "__main__":
    main()
