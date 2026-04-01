# GUI Architecture
The goal is to keep the UI a simple presentation layer on top of the game/runtime layers. User inputs are emitted from widgets, translated by presenters into `SessionManager` calls, and widgets are updated from app signals.

## Overview
- The Qt6 front end is a thin shell around `tengen::app::SessionManager` and the underlying runtime/core layers.
- The board view is a Qt `QWidget` (`BoardWidget`) that uses `QPainter` to render the grid and stone textures directly onto the widget.

## Responsibilities
- `tengen::gui::MainWindow` builds the Qt layout and emits connect/host/shutdown requests. It does not handle game state directly.
- `tengen::gui::BoardWidget` converts input into `BoardWidgetEvent`s and renders the board through `BoardRenderer`.
- The board drawing itself remains inside `tengen::gui::BoardRenderer`, so renderer changes stay isolated from the rest of the UI.

## Threading and Updates
- The core game can run on a dedicated `std::thread`; the Qt event loop stays on the main thread.
- UI-to-runtime: widget signals are handled by presenters, which forward actions to `SessionManager`.
- Runtime-to-UI: presenters receive app signals and repost widget updates onto the Qt thread via `QMetaObject::invokeMethod`.

## Extensibility
- The side tabs and footer are empty hooks—drop in history, chat, or status widgets without touching the rendering path.
- Because the renderer is pure Qt painting logic, swapping to other Qt drawing approaches (e.g., `QGraphicsScene`) remains isolated to `BoardRenderer` and `BoardWidget`.
- Keep business logic inside the runtime/core layers so the GUI remains a presentation layer only.
