# Core Library (`gameCore`)

Rules engine for Go. For the conceptual picture, see [docs/Core.md](../../../docs/Core.md) — this is the companion for people changing code in here. Depends on `game::model` only; don't let Qt or networking leak in.

## Concurrency

`pushEvent()` and the subscribe/unsubscribe calls are the only synchronized entry points into `Game`. `boardSize()` reads the position that `run()` replaces on the game thread, so don't assume it is safe to poll from outside the game thread. Whether the game is still on lives in `GameState` and reaches other threads only through `GameDelta::gameActive`; `run()` keeps handling events after the game ended, until a `ShutdownEvent`.

`EventHub::signal()`/`signalDelta()` hold `m_listenerMutex` for the entire dispatch loop and call listeners synchronously and inline — in practice on the game thread, since `Game::handleEvent` calls straight into the hub. A slow listener stalls `Game::run()` itself, and a listener that calls `subscribe`/`unsubscribe` back into the same hub from inside its own callback will deadlock, since the mutex isn't recursive. Keep callbacks fast and non-reentrant.

## Where to look

- `game.*` — event loop and orchestration.
- `gameState.*` — rules state of one game: position, turn order, ko history, game end.
- `gameRules.*` — rule presets and options.
- `moveChecker.*` — legality, captures, liberties.
- `position.*` — `GamePosition`.
- `eventHub.*`, `IGameSignalListener.hpp`, `IGameStateListener.hpp` — the notification system.
- `zobristHash.hpp` / `IZobristHash.hpp` — hashing.
- `sgfHandler.*`, `serializer.*` — coordinate and text-board conversion utilities.
