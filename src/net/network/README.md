# Game Networking Layer (`netNetwork`)

This module sits between the raw TCP transport (`netCore`) and the game logic.
It turns typed game events into wire messages and back, and it manages server/client roles.

## Big Picture

- **Client**: `network::Client` wraps `core::TcpClient` and handles the protocol; the transport brings its own IO thread.
- **Server**: `network::Server` wraps `core::TcpServer` and exposes a clean event callback.
- **Events**: All wire messages are defined in `nwEvents.hpp` and serialized as JSON.
- **Sessions**: `SessionManager` maps `ConnectionId` <-> `SessionId` and tracks seats.

## Design Choices

- **Small protocol**: JSON is used for clarity and quick iteration. May be improved in the future.
- **Thread split**: server processing is on its own thread; the client handles messages on the IO thread of its `TcpClient` and has no thread of its own.

## Where To Look

- `src/net/network/include/network/nwEvents.hpp` for the protocol types.
- `src/net/network/nwEvents.cpp` for serialization/parsing.
- `src/net/network/server.*` for the server wrapper and event forwarding.
- `src/net/network/client.*` for the client wrapper and the handshake.
