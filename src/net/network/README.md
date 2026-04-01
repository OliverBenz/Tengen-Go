# Game Networking Layer (`netNetwork`)

This module sits between the raw TCP transport (`netCore`) and the game logic.
It turns typed game events into wire messages and back, and it manages server/client roles.

## Big Picture

- **Client**: `network::Client` wraps `network::TcpClient` and runs a read thread.
- **Server**: `network::Server` wraps `network::TcpServer` and exposes a clean event callback.
- **Events**: All wire messages are defined in `nwEvents.hpp` and serialized as JSON.
- **Sessions**: `SessionManager` maps `ConnectionId` <-> `SessionId` and tracks seats.

## Design Choices

- **Small protocol**: JSON is used for clarity and quick iteration. May be improved in the future.
- **Thread split**: server processing is on its own thread; client has a blocking read thread.

## Where To Look

- `src/net/network/include/network/nwEvents.hpp` for the protocol types.
- `src/net/network/nwEvents.cpp` for serialization/parsing.
- `src/net/network/server.*` for the server wrapper and event forwarding.
- `src/net/network/client.*` for the client wrapper and read loop.
