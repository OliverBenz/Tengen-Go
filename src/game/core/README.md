# Core Library (`gameCore`)

Rules engine for Go. For the conceptual picture, see [docs/Core.md](../../../docs/Core.md) — this is the companion for people changing code in here. Depends on `game::model` only; don't let Qt or networking leak in.

## Concurrency

`pushEvent()` and the subscribe/unsubscribe calls are the only synchronized entry points into `Game`. The `GameState` is only touched on the game thread; other threads learn about the game through the listeners alone, never by reading it.

A game is constructed with its `GameConfig` and refuses moves until a `StartEvent` is handled, which sends `onGameStart()` with the config. Two passes or a resignation end it: `GameState` records the `GameResult`, and `onGameEnd()` follows right after the delta of that last move. `run()` keeps handling events before the start and after the end, until a `ShutdownEvent`. Subscribe state listeners before the `StartEvent`; a late one never gets `onGameStart()`.

`EventHub::signal()`/`signalStart()`/`signalDelta()`/`signalEnd()` hold `m_listenerMutex` for the entire dispatch loop and call listeners synchronously and inline — in practice on the game thread, since `Game::handleEvent` calls straight into the hub. A slow listener stalls `Game::run()` itself, and a listener that calls `subscribe`/`unsubscribe` back into the same hub from inside its own callback will deadlock, since the mutex isn't recursive. Keep callbacks fast and non-reentrant.

## Where to look

- `game.*` — event loop and orchestration.
- `gameState.*` — rules state of one game: turn order, game end. Runs every move through the two below.
- `moveChecker.*` — board mechanics: captures, liberties, suicide. Knows nothing about the game history.
- `positionHistory.*` — earlier positions and the ko rule.
- `position.*` — `GamePosition`.
- `eventHub.*`, `IGameSignalListener.hpp`, `IGameStateListener.hpp` — the notification system.
- `zobristHash.hpp` / `IZobristHash.hpp` — hashing.
- `sgfHandler.*`, `serializer.*` — coordinate and text-board conversion utilities.
