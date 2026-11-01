# Tengen (天元)

Tengen (天元) is the center point of a Go board. This project aims to be the center point between physical and digital Go.

Tengen is a modular C++ system that combines:

- Go game logic and a rules engine
- TCP networking
- A GUI client
- Computer vision that detects moves on a real board
- Robotic arm control that places stones on a real board

The whole project is still very much work in progress. The current focus is the image detection and a robotic arm that mirrors the opponent's move on the physical board.
It is kept modular, so you can take whatever parts are useful to you.

## Motivation

I don't enjoy playing on the computer and don't have Go-interested people near me.
So let's replace the opponent with a robotic arm and play other people online, but over a real board.

The final goal is a full robotic Go set: a documented parts list for the hardware, and the software in this repository.
You buy the parts at the best available price, enjoy assembling everything, install this software, and get local and online games, puzzles, and training against bots.
All open source, so you can tinker as you like.

## Status

| Area             | Status                 | Notes                                                                                            |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| Game Logic       | Working                | Core data structures, rules, move validation, and deltas are implemented.                        |
| Bot Games        | Working / In Progress  | Bot games against GNU Go and KataGo follow the selected rules; some edge cases are still open.   |
| Networking       | Working / In Progress  | TCP transport and game protocol exist; reconnect and some session features are still incomplete. |
| GUI Application  | Working / In Progress  | Qt client and standalone server exist; the application is still under active development.        |
| Board Detection  | Working / Experimental | Board, grid, and stone detection exist, but still rely on a controlled setup.                    |
| Board Perception | In Progress            | Setup/orientation logic exists; the live board detection loop is not finished yet.               |
| Robot Arm        | Planned                | The long-term goal is documented, but nothing is in the repository yet.                          |

## Components

| Name             | Description                                                                          |
| ---------------- | ------------------------------------------------------------------------------------ |
| gameModel        | Core data structures of the game.                                                    |
| gameCore         | Game rules, board state validation, deltas, and move handling.                       |
| gameEngine       | Drives bot engines (GNU Go, KataGo) as separate processes over the Go Text Protocol. |
| gameGui          | Qt 6 widgets built on gameModel.                                                     |
| gameRuntime      | Application logic, connecting the game with vision and networking.                   |
| netCore          | Low-level TCP transport, framing, and connection management.                         |
| netNetwork       | Game protocol and client/server session handling, built on netCore.                  |
| visionCore       | Board, grid, and stone detection with OpenCV.                                        |
| visionPerception | Game-specific image detection, built on visionCore.                                  |
| tengen           | The application, built on gameRuntime and gameGui.                                   |

The executables mainly handle input and output and talk to the runtime and core libraries.
For example, `tengen` renders what the runtime reports and forwards user input through presenters and the session manager.

Besides `tengen` there is a standalone `server`, and two development tools: `visionTuner` visualizes the steps of the vision pipeline, and `boardViewer` renders board states from different file formats.
Tests live in `tests/`, one project per component and kind, e.g. `gameCore.unit` or `visionCore.integration`. Not every component has detailed tests yet.

### External Dependencies

| Name          | Used for                                   |
| ------------- | ------------------------------------------ |
| Qt 6          | GUI and audio.                             |
| OpenCV 5      | Image processing in the vision components. |
| Asio          | Networking.                                |
| nlohmann_json | JSON parsing.                              |
| Logger        | Logging.                                   |
| GTest         | Unit tests.                                |
| CMake files   | Shared project settings (`lib/cmake`).     |

## Building

The project is configured via [`CMakePresets.json`](CMakePresets.json): `Linux64-Debug`/`Linux64-Release`/`Linux64-RelWithDebInfo` (Ninja) on Linux, and `Win64` (Visual Studio 17 2022, Debug/Release/RelWithDebInfo) on Windows.

```
cmake --preset Linux64-Debug && cmake --build --preset Linux64-Debug
```

Qt 6 and OpenCV are required. On Linux they come from the system package manager.
On Windows, [`scripts/win/1_setup_deps.py`](scripts/win/1_setup_deps.py) installs both. See the [Compilation Instructions](docs/Compilation.md) for details.

The Visual Studio generator can't write `compile_commands.json` for clangd.
On Windows, run [`scripts/win/2_configure-clangd.ps1`](scripts/win/2_configure-clangd.ps1) instead: it configures the `Win64-Ninja` preset (MSVC + Ninja) from an x64 VS developer shell and copies the file to the project root.

## Bot Games

Bot games are played against [GNU Go](https://www.gnu.org/software/gnugo/) or [KataGo](https://github.com/lightvector/KataGo).
They run as separate processes and are not part of this repository, so you have to provide them yourself.

Tengen looks for them in an `engine` folder, first in the user's data folder, then next to its own executable (for a Windows debug build, that is `build/Win64/out/bin/x64/Debug/`).
Each engine comes from the first folder that holds all of its files, and engines in the data folder serve every build.

| OS      | Data folder                                   |
| ------- | --------------------------------------------- |
| Linux   | `~/.local/share/tengen/engine`                |
| Windows | `%LOCALAPPDATA%\tengen\engine`                |
| macOS   | `~/Library/Application Support/tengen/engine` |

```
engine/
├── gnugo/
│   └── gnugo(.exe)
└── katago/
    ├── katago(.exe)
    ├── model.bin.gz
    ├── human_model.bin.gz
    └── gtp.cfg
```

The bot dialog lists both engines. An engine whose files are not all there is greyed out. The dialog checks again every time it opens, so an engine installed while Tengen runs shows up right away.

### GNU Go

On Windows, download a GNU Go 3.8 build and copy its whole folder into `engine/gnugo/`, not just `gnugo.exe`: builds like the Cygwin one need their DLLs next to the executable.

On Linux, install GNU Go through your package manager and link it into place:

```
mkdir -p ~/.local/share/tengen/engine/gnugo
ln -s "$(command -v gnugo)" ~/.local/share/tengen/engine/gnugo/gnugo
```

GNU Go's strength is a level from 1 to 10, and that is what the bot dialog offers.
The levels are not ranks: GNU Go plays around 5k to 8k at level 10, and the lower levels play weaker by an amount nobody has measured.

### KataGo

Copy KataGo's whole folder into `engine/katago/`, then add:

- a KataGo network, as `model.bin.gz`
- the human SL network `b18c384nbt-humanv0.bin.gz` from [katagotraining.org's extra networks](https://katagotraining.org/extra_networks/), as `human_model.bin.gz`
- a GTP config made for the human SL network, like KataGo's `gtp_human5k_example.cfg`, as `gtp.cfg`

KataGo imitates a human of the rank you pick in the bot dialog, from 20k to 3d.

## Documentation

- [Compilation](docs/Compilation.md): installing dependencies and building
- [Release](docs/Release.md): publishing a release through a version tag
- [Core](docs/Core.md): core rules and logic overview
- [GUI](docs/GUI.md): GUI architecture and rendering notes
- [Networking](docs/Networking.md): higher-level networking notes

Per component:

- [game::model](src/game/model/README.md): modelling the data of a real game
- [game::core](src/game/core/README.md): rules, game loop, deltas, and move validation
- [net::core](src/net/core/README.md): network transport and framing
- [net::gameNet](src/net/network/README.md): protocol, client/server wrappers, and session mapping
- [vision::core](src/vision/core/README.md): board, grid, and stone detection pipeline
- [vision::perception](src/vision/perception/README.md): mapping detected stones onto game coordinates

## License

Tengen is meant to be a free, open-source product for Go enthusiasts and hobbyists, and derived versions should stay open to everyone.
Components that are useful on their own are licensed more permissively, so you can use them in your own projects.

| Part                                                                           | License                                                                      |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Tengen, everything not listed below                                            | [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt)                                  |
| netCore (`src/net/core`)                                                       | [Apache-2.0](LICENSES/Apache-2.0.txt)                                        |
| visionCore (`src/vision/core`)                                                 | [Apache-2.0](LICENSES/Apache-2.0.txt)                                        |
| Third-party images, board textures, and own sounds in `src/game/gui/resources` | Their own licenses, listed in its [README](src/game/gui/resources/README.md) |

The license of every file is listed in [`REUSE.toml`](REUSE.toml), the license texts are in [`LICENSES/`](LICENSES).
[`NOTICES.md`](NOTICES.md) lists the licenses of everything that ships with Tengen, including Qt and OpenCV, and is installed with it.

Copyright (C) 2024 Oliver Benz.
