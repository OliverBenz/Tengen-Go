# Tengen-Go AGENTS.md

## Project

C++20 Go (board game) engine with TCP networking, Qt 6 GUI, and OpenCV-based camera board recognition.
Bridges physical and digital play: detect opponent's moves on a real board and mirror them digitally (robotic arm future).

## Prerequisites (agents often miss these)

```sh
# Dependencies are git submodules, all images (GUI resources, vision test photos, training data, docs) are Git LFS.
# Needs git-lfs installed. Run before any build step, otherwise images are LFS pointer text and builds/tests fail.
scripts/1_fetch.sh   # = git submodule update --init --recursive && git lfs pull

# Linux: cmake, ninja, gcc/clang, Qt 6.8+, OpenCV 5 (system package)
sudo pacman -S git-lfs qt6-base qt6-multimedia opencv   # example: Arch

# Windows: VS 2022 C++ workload + Python 3.11, then:
python scripts/win/1_setup_deps.py
```

`flake.nix` is incomplete (OpenCV5 not in nixpkgs). Use system deps instead.

## Build & Test

Pick a preset from `CMakePresets.json`; output goes to `build/<preset>`. On Linux the default dev target is `Linux64-Debug`:

```sh
cmake --preset Linux64-Debug                     # configure once
cmake --build --preset Linux64-Debug --parallel   # build
ctest  --test-dir build/Linux64-Debug --output-on-failure --parallel $(nproc)  # run all tests
ctest -R <regex>                                  # filter: e.g. `game.core` or `visionCore.unit`
# Single case:
ctest -R game.core.unit --gtest_filter="BoardTest.InitSize"
```

## Tooling scripts (`scripts/`)

| Script              | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `1_fetch.sh`        | Submodules + `git lfs pull`. CI does the same via `submodules: recursive` + `lfs: true`. |
| `format.sh`         | clang-format all C++ tracked files. Use `--check` for CI. |
| `format.sh --check` | Dry-run; exits 1 on unformatted files.                    |
| `tidy.sh [path]`    | Runs clang-tidy over `src/` and `tests/`. Defaults to `Linux64-Debug` build dir. Set `BUILD_DIR=presets/...`. Accepts `--strict` for CI (warnings-as-errors). Internally strips `-mno-direct-extern-access` from compile_commands.json before running — do not try to replicate this manually. |

## Dependencies (`lib/`)

External deps are pulled in via CMake scripts: `asio`, `gtest`, `logger`, `nlohmann_json`.
`dependencies.cmake` handles Qt 6 + OpenCV 5 find-module lookup.
All libraries build into the standard `build/<preset>/lib/` tree.

## Testing (`tests/`)

Target naming follows `<module>.<kind>` or `<module>.<kind>.<detail>`:

```
game/core.unit          game/engine.unit
net/core.integration    net/network.unit   net/network.integration
vision/core.unit        vision/core.integration  vision/core.analysis
vision/perception.integration
```

- Tests run in parallel; keep them independent.
- Some vision tests are disabled — a green suite does not cover everything there (as noted in the tests themselves).

## Architecture (modules)

```
src/game/              src/net/              src/vision/           src/apps/
├── model/             ├── core/            ├── core/             ├── tengen/    ← client GUI app
├── core/              ├── network/         ├── perception/       ├── server/    ← standalone server
├── engine/          (C++)                (C++)                 └── engines/   ← bot communication
├── runtime/
└── gui/
tools/                 3rdparty/             lib/                  docs/
(dev tools)          (don't modify!)    (dep cmake scripts)   (compilation, engine, etc.)
```

Libraries → core model/data → engine rules -> networking/serialization -> runtime/orchestration -> gui.
Executables (`tengen`, `server`) are thin: input/output + presentation. Business logic lives in libraries.

`tools/` holds dev tools (`visionTuner`, `boardViewer`, `manualSetup`, `geometryLabeler`, `stoneClassifier` in Python). The Qt tools follow the same presenter/view rules. `geometryLabeler` writes the geometry `.json` and `board.txt` described in `tests/vision/resources/README.md`; `stoneClassifier` trains on those files, see its README and the CNN primer in `docs/stoneClassifier/`.
Images for tests, training and docs are Git LFS: new image paths must be covered by `.gitattributes`.

### Key constraints

- **GUI = presenter pattern.** View layer is thin: no state (game, runtime or workflow, e.g. selections or counts) and no decision logic. It shows what the presenter tells it and emits user input as signals. Presenters get collaborators (other views/dialogs, I/O) injected; the composition root is `main`.
- **No global state; dependency injection preferred.**
- **`include/<project>/` for public headers (future); `src/` for private today.** New code should follow this split.

## Licensing (REUSE)

The repo uses [REUSE](https://reuse.software). File-level licenses are in [`REUSE.toml`], texts in `LICENSES/`, package notices in `NOTICES.md`.
Valid with `reuse lint`. **Never remove or change license headers.** When adding files, ask about licensing if unclear.

| Part                                    | License      |
| --------------------------------------- | ------------ |
| Everything under `src/game/gui/resources`  | See that dir's README |
| Remaining Tengen code                   | AGPL-3.0-only |
| `src/net/core` (netCore)                | Apache-2.0   |
| `src/vision/core` (visionCore)           | Apache-2.0   |

## Code conventions (summary — see `.clang-format` / `.clang-tidy` for full config)

- **Ownership:** `unique_ptr` first, `shared_ptr` only for genuine shared ownership. No `new`/`delete`/`malloc`.
- **Class layout:** public section on top, private at the bottom.
- **Naming:** descriptive but concise; self-documenting code over comments.
- **Comments:** `//!` / `//!<` for documented symbols, `//` for general notes. Short and only when important.

## CMake rules

- Use `PRIVATE` visibility first; set `PUBLIC` only when a transitive consumer needs it.
- Do not change existing visibility settings (assume intentional).
- Avoid transitive dependencies; keep implementation details private.

## Docs

See `docs/` for deeper guidance beyond what belongs in an agent instruction file:
[`Compilation.md`](docs/Compilation.md) · [`Core.md`](docs/Core.md) · [`GUI.md`](docs/GUI.md) · [`Networking.md`](docs/Networking.md) · [`Engine.md`](docs/Engine.md) · [`Release.md`](docs/Release.md) · [`stoneClassifier/`](docs/stoneClassifier/) (CNN primer, PDF)

## General rules

- Only make changes when explicitly asked. Do not commit unless requested.
- When requirements are unclear, ask the developer before deciding.
