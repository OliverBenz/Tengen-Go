# Stone classifier

A small CNN that classifies a board intersection as empty, black or white.
A CNN primer is in [`docs/stoneClassifier/cnn_course.pdf`](../../docs/stoneClassifier/cnn_course.pdf).

## Workflow

1. **Label photos.** Each photo needs a `<photo>.json` with its grid corners and a dotBW layout, either `<photo>.txt` or the folder's `board.txt`. Click them with `tools/geometryLabeler`, formats in [`tests/vision/resources/README.md`](../../tests/vision/resources/README.md).
2. **Extract** a frozen dataset version: `extract_patches.py <data>/stones_vN`
3. **Check** it by eye: `stones_vN/check/` has every rectified board with its crops marked, and contact sheets of all patches per class.
4. **Train** one model: `train.py <data>/stones_vN/stones.npz <data>/runs/<name>`,
   or **compare options** in parallel: `sweep.py <stones.npz> <data>/runs/<sweep> --options ... --splits ...`
5. **Evaluate** and look at the mistakes: `evaluate.py <run>/model.pt <stones.npz> <run>/eval`
6. **Export** the best run for the vision library: `export_onnx.py <run>/model.pt` writes
   `src/vision/core/models/stoneNet.onnx` and `stoneNet.json`, which says how to prepare the input and where the model came from.
   Only the current model lives in the repo; every run stays in the data directory.
7. **Try it live**: `live_preview.py` outlines the stones on a camera image, after clicking the 4 grid corners.

Every output directory must not exist yet, so a dataset version or run is never overwritten.
Compare training options on several seeds: a single run varies by several stones.
The device and thread count change results slightly too, so keep them equal within a comparison.

## Layout

```
extract_patches.py   script: photos → stones.npz, manifest.json, check/
train.py             script: training loop → model.pt, run.json
evaluate.py          script: confusion matrix and contact sheets of the mistakes
sweep.py             script: options × test splits × seeds as parallel train.py runs, one comparison table
export_onnx.py       script: model.pt → src/vision/core/models/stoneNet.onnx + .json, checked against PyTorch
live_preview.py      script: camera preview with the detected stones outlined
stone_classifier/
├── provenance.py    file hashes and git commit for the records
├── preparation/
│   ├── corpus.py    find the labelled photos, load their geometry and layout
│   ├── rectify.py   warp a photo onto the canonical grid, turn its layout to match, cut patches
│   └── preview.py   renderings for checking by eye: boards, contact sheets
└── learning/
    ├── dataset.py   stones.npz → network input: crop, augmentation
    ├── model.py     the CNN, saving and loading it
    └── metrics.py   predictions, confusion matrix, recall
training_data/       labelled photos (Git LFS), same format as tests/vision/resources
```

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The default `torch` wheel includes CUDA. On the GPU, `train.py` switches to deterministic algorithms, so a run is
reproducible there as well.
