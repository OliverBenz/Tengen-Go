# Tengen (天元)

A modular Go system combining game logic, networking, and real-world board perception.

Tengen (天元) refers to the center point of a Go board. This project aims to be the center point between physical and digital Go.

Tengen is a modular C++ system that combines:

- Go game logic and rules engine
- TCP networking
- GUI client
- Computer vision for detecting moves on a real board
- Robotic arm control to place stones on a real board

The goal is to seamlessly integrate physical gameplay with digital systems.
Challenging opponents remotely while playing over a real board instead of staring at a screen.

The whole project is still very much work in progress. Current focus lies on the image detection system as well as a robotic arm that can mirror the opponents move on the physical board.
The project is aimed to be kept modular so you can easily just take whatever parts are useful to you.

## Motivation

The motivation for this project is simple.
I don't enjoy playing on the computer and don't have Go-interested people near me.
So let's replace the opponent with a robotic arm and play other people online but over the board.

## Current Status

| Area             | Status                 | Notes                                                                                            |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| gameModel/Core   | Working                | Core data structures, rules, move validation, and deltas are implemented.                        |
| netCore/Network  | Working / In Progress  | TCP transport and game protocol exist; reconnect and some session features are still incomplete. |
| GUI Application  | Working / In Progress  | Qt client and standalone server exist; the application is still under active development.        |
| visionCore       | Working / Experimental | Board, grid, and stone detection exist, but still rely on a controlled setup.                    |
| visionPerception | In Progress            | Setup/orientation logic exists; the live board detection loop is not finished yet.               |
| Robot Arm        | Planned                | The long-term goal is documented, but this is not shipped in the repository yet.                 |

### Goal

The final goal is to have a full robotic Go set.
We may document a parts list for the hardware and provide the software here.
A user may then purchase this hardware at the best available price and experience the fun of assembling everything.
Finally flashing this software to get access to local and online games, puzzles, and training against bots.
All open source so you can tinker around as you like.

## Components

### Internal Components

| Name             | Description                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| gameModel        | Library specifying the core data structures for the game.                                               |
| gameCore         | Library for game rules, board state validation, deltas, and move handling.                              |
| gameEngine       | Library for driving bot engines (GNU Go, KataGo) as separate processes over the Go Text Protocol.       |
| gameGui          | Library for QT6 graphical user elements built on the gameModel.                                         |
| netCore          | Library for low-level TCP transport, framing, and connection management.                                |
| netNetwork       | Library for the game/network protocol and client/server session handling. Building on netCore.          |
| visionCore       | Library for board, grid, and stone detection using OpenCV.                                              |
| visionPerception | Library for the game specific image detection. Building on visionCore.                                  |
| gameRuntime      | Library defining the application logic. Connecting the core game with vision algorithms and networking. |
| tengen           | The final application built on the runtime and GUI libraries.                                           |

Including a [ComponentName].GTest project which should be managed for each component.
As the whole project is still very much work-in-progress, detailed testing is not yet present for each module.

The executables mainly specify IO handling and communicate with the runtime/core libraries.
For example, `tengen` renders information from the runtime layer and forwards user input through presenters and the session manager.
We also provide a standalone `server` and tools for development purposes - like the `visionTuner` for debugging and visualizing the vision pipeline steps and the `boardViewer` which aims to allow to render go board states from different file formats.
These are currently very basic and to be extended as required.

### External Components

| Name   | Description                        |
| ------ | ---------------------------------- |
| CMake  | Collection of CMake files.         |
| Logger | Library for logging functionality. |
| Asio   | Library for networking support.    |
| GTest  | Google unit testing library.       |

## Building

The project is configured via [`CMakePresets.json`](CMakePresets.json): `Win64` (Visual Studio 17 2022, Debug/Release/RelWithDebInfo) on Windows, and `Linux64-Debug`/`Linux64-Release`/`Linux64-RelWithDebInfo` (Ninja) on Linux.

```
cmake --preset Win64 && cmake --build --preset Win64-Debug
```

OpenCV has no package manager on Windows, so `OpenCV_DIR` must point at the OpenCV build directory containing `OpenCVConfig.cmake`.
The `Win64` preset currently hardcodes this to `C:/opencv/build`; if your install lives elsewhere, edit that path or override it in a local `CMakeUserPresets.json`.
On Linux, OpenCV is expected to come from the system package manager, so no manual step is needed there.

## Bot Games

Bot games are played against [GNU Go](https://www.gnu.org/software/gnugo/) or [KataGo](https://github.com/lightvector/KataGo).
They run as separate processes and are not part of this repository, so you have to provide them yourself.
Tengen looks for them in an `engine` folder, first in the user's data folder, then next to its own executable (for a Windows debug build, that is `build/Win64/out/bin/x64/Debug/`).
Each engine comes from the first folder that holds all of its files. Engines in the data folder serve every build.

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

The bot dialog lists both engines. One whose files are not all there is greyed out, and the dialog looks again every time it opens, so an engine installed while Tengen runs shows up right away.

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

## General Documentation

- [Core](docs/Core.md) — core rules/logic overview
- [GUI](docs/GUI.md) — GUI architecture and rendering notes
- [Networking](docs/Networking.md) — higher-level networking notes

## Technical Documentation

- [gameCore](src/game/core/README.md) — Core rules, game loop, deltas, and move validation
- [netCore](src/net/core/README.md) — Network transport and framing details
- [netNetwork](src/net/network/README.md) — Protocol, client/server wrappers, and session mapping
- [visionCore](src/vision/core/README.md) — Board, grid, and stone detection pipeline
- [visionPerception](src/vision/perception/README.md) — Mapping detected stones onto game coordinates

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-or-later). See `LICENSE`.

Copyright (C) 2024 Oliver Benz.
