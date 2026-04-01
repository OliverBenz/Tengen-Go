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

### Goal
The final goal is to have a full robotic Go set.
We may document a parts list for the hardware and provide the software here.
A user may then purchase this hardware at the best available price and experience the fun of assembling everything.
Finally flashing this software to get access to local and online games, puzzles, and training against bots.
All open source so you can tinker around as you like.

## Components
### Internal Components
Name        | Description 
------------|------------
gameCore    | Library for game rules, board state validation, deltas, and move handling.
netCore     | Library for low-level TCP transport, framing, and connection management.
netNetwork  | Library for the game/network protocol and client/server session handling.
visionCore  | Library for board, grid, and stone detection using OpenCV.
tengen      | Qt GUI application built on the runtime and GUI libraries.

Including a [ComponentName].GTest project for each component.

The executables mainly specify IO handling and communicate with the runtime/core libraries.
For example, `tengen` renders information from the runtime layer and forwards user input through presenters and the session manager.


### External Components
Name   | Description
-------|------------
CMake  | Collection of CMake files.
Logger | Library for logging functionality.
Asio   | Library for networking support.
GTest  | Google unit testing library.

## General Documentation
- [General](docs/Documentation.md) — entry point and general notes
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
