# Networking

## Current Architecture (what actually exists)

We split networking into two layers:

- **netCore**: pure transport (TCP, framing, connection management).
- **netNetwork**: application protocol (game events, server/client roles, session IDs).

The server is authoritative. Clients send intent; server sends facts.

### Components

- **Server**: `network::Server` running in the server executable. Forwards client events to the game loop and broadcasts deltas/updates to clients.
- **Clients**: `network::Client` used by the GUI. Sends move intents and applies server updates.
- **Observers**: same client type, just a different seat.

### Data Flow (happy path)

1) Client sends an intent (`ClientPutStone`, `ClientPass`, `ClientResign`).
2) Server forwards the intent to the core game loop.
3) Game validates and emits an update (`GameDelta`).
4) Server turns that into a `ServerDelta` and broadcasts to all clients.
5) Clients apply the delta locally and update UI.

This structure enforces a clear separation between layers and responsibilities.

### Layers in code

**netCore**
- `TcpServer` owns the accept loop and active `Connection` objects.
- `Connection` does async read/write with a size‑prefixed frame.
- `TcpClient` is the synchronous client used by the GUI and tests.

**netNetwork**
- `Client` and `Server` wrap the transport with a small protocol in `nwEvents`.
- `SessionManager` (server‑side) maps connections to seats and handles disconnects.
- `ServerDelta` is the core update payload; clients treat it as the single source of truth.

### Message framing

Messages are length‑prefixed in `netCore`:

1) `BasicMessageHeader` contains payload size (network byte order).
2) Then `payload_size` bytes of payload.

`netNetwork` serializes/deserializes the payload into typed events.

### Notes

- The server does not trust clients. Clients only *request* moves.
- Clients keep a local shadow state for rendering, but server data wins.
- Observers are just clients with an observer seat.

### Where to look

- Transport: `src/net/core/README.md`
- Protocol: `src/net/network/include/network/nwEvents.hpp`
- Server wiring: `src/net/network/server.cpp`, `src/game/runtime/gameServer.cpp`
- Client wiring: `src/net/network/client.cpp`, `src/game/runtime/sessionManager.cpp`
