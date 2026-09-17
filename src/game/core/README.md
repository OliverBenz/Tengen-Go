# Core Library (`gameCore`)

Rules engine for Go. For the conceptual picture, see [docs/Core.md](../../../docs/Core.md) — this is the companion for people changing code in here. Depends on `game::model` only; don't let Qt or networking leak in.

## Concurrency

`pushEvent()` and the subscribe/unsubscribe calls are the only synchronized entry points into `Game`. `isActive()` and `boardSize()` read plain fields that `run()` writes on the game thread — `GameServer` already calls `isActive()` from the network thread on every incoming move. That's a real data race, just one that's been harmless in practice so far; don't assume either getter is safe to poll from outside the game thread.

`EventHub::signal()`/`signalDelta()` hold `m_listenerMutex` for the entire dispatch loop and call listeners synchronously and inline — in practice on the game thread, since `Game::handleEvent` calls straight into the hub. A slow listener stalls `Game::run()` itself, and a listener that calls `subscribe`/`unsubscribe` back into the same hub from inside its own callback will deadlock, since the mutex isn't recursive. Keep callbacks fast and non-reentrant.

## Where to look

- `game.*` — event loop and orchestration.
- `moveChecker.*` — legality, captures, liberties.
- `position.*` — `GamePosition`.
- `eventHub.*`, `IGameSignalListener.hpp`, `IGameStateListener.hpp` — the notification system.
- `zobristHash.hpp` / `IZobristHash.hpp` — hashing.
- `sgfHandler.*`, `serializer.*` — coordinate and text-board conversion utilities.
