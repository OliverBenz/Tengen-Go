"""Run training options × test splits × seeds as parallel train.py processes, then compare them in one table.

  sweep.py <stones.npz> <out> --options base="" wide="--width 32" \\
           --splits hard="angled_hard_lighting" board2="board_2 board_2_lamp" --seeds 0 1 2 --jobs 16 --threads 2

Every run writes <out>/<option>__<split>__s<seed>/, see train.py. The table is printed and saved as <out>/summary.txt.
All runs of a sweep use the same device and thread count, since both change the results slightly.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train.py"


def main():
    parser = make_parser()
    args = parser.parse_args()
    options = parse_pairs(parser, args.options)
    splits = parse_pairs(parser, args.splits)

    # 1. One train.py command per combination of option, split and seed.
    args.out.mkdir(parents=True)  # Fails if it exists: a written sweep is frozen.
    commands = {}
    for (option, option_args), (split, sets), seed in product(options.items(), splits.items(), args.seeds):
        name = f"{option}__{split}__s{seed}"
        commands[name] = [
            sys.executable, str(TRAIN), str(args.dataset), str(args.out / name),
            "--seed", str(seed), "--test-sets", *shlex.split(sets),
            "--device", args.device, "--threads", str(args.threads), *shlex.split(option_args),
        ]

    # 2. Run them, `jobs` at a time.
    run_all(commands, args.jobs)

    # 3. Compare the options per split.
    summary = summarise(args.out, options, splits, args.seeds)
    print(summary, end="")
    (args.out / "summary.txt").write_text(summary)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="stones.npz of an extracted dataset version")
    parser.add_argument("out", type=Path, help="new output directory for all runs of this sweep")
    parser.add_argument("--options", nargs="+", required=True, help='name="train.py options", e.g. wide="--width 32"')
    parser.add_argument("--splits", nargs="+", required=True, help='name="test sets", e.g. hard="angled_hard_lighting"')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=16, help="runs at the same time")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="all runs share it")
    return parser


def parse_pairs(parser: argparse.ArgumentParser, pairs: list[str]) -> dict[str, str]:
    """`name=value` arguments as a dict, in the given order."""
    parsed = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name or "__" in name:
            parser.error(f"expected name=value with a name free of '__', got {pair!r}")
        parsed[name] = value
    return parsed


def run_all(commands: dict[str, list[str]], jobs: int):
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(subprocess.run, command, capture_output=True, text=True): name
                   for name, command in commands.items()}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = "ok" if result.returncode == 0 else f"FAILED\n{result.stderr[-2000:]}"
            elapsed = time.monotonic() - start
            print(f"[{done}/{len(commands)}] {elapsed:5.0f} s  {futures[future]}  {status}", flush=True)


def summarise(out: Path, options: dict[str, str], splits: dict[str, str], seeds: list[int]) -> str:
    """Per split, one row per option. Error columns are summed over the seeds; 'per seed' columns list each run."""
    header = (f"{'option':<14}{'test errors per seed':>22}{'false stones':>14}{'missed stones':>15}"
              f"{'black<->white':>15}{'train errors per seed':>23}{'s per run':>11}")
    lines = []
    for split in splits:
        lines += ["", f"== test {split}: {splits[split]}", header]
        for option in options:
            paths = [out / f"{option}__{split}__s{seed}" / "run.json" for seed in seeds]
            records = [json.loads(path.read_text()) for path in paths if path.exists()]
            if len(records) < len(seeds):
                lines.append(f"{option:<14}  {len(seeds) - len(records)} of {len(seeds)} runs failed")
                continue
            lines.append(summary_row(option, records))
    return "\n".join(lines) + "\n"


def summary_row(option: str, records: list[dict]) -> str:
    """Classes are (empty, black, white). A confusion matrix has rows for the true, columns for the predicted class."""
    test = [record["final_confusion"] for record in records]
    train = [record["final_train_confusion"] for record in records]
    false_stones = sum(c[0][1] + c[0][2] for c in test)
    missed_stones = sum(c[1][0] + c[2][0] for c in test)
    colour_swaps = sum(c[1][2] + c[2][1] for c in test)
    seconds = sum(record["seconds"] for record in records) / len(records)
    return (f"{option:<14}{per_seed(test):>22}{false_stones:>14}{missed_stones:>15}{colour_swaps:>15}"
            f"{per_seed(train):>23}{seconds:>11.0f}")


def per_seed(confusions: list[list[list[int]]]) -> str:
    errors = [sum(row[j] for i, row in enumerate(c) for j in range(len(row)) if i != j) for c in confusions]
    return " / ".join(map(str, errors))


if __name__ == "__main__":
    main()
