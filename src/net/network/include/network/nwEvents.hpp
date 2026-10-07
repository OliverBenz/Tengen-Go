#pragma once

#include "model/coordinate.hpp"
#include "model/gameConfig.hpp"
#include "model/gameDelta.hpp"
#include "model/gameResult.hpp"
#include "model/player.hpp"
#include "network/types.hpp"

#include <optional>
#include <string>
#include <variant>

namespace tengen::network {

// Client Network Events (client -> server)
struct ClientPutStone {
	Coord c;
};
struct ClientPass {};
struct ClientResign {};
struct ClientChat {
	std::string message;
};

// Server Events (server -> client)
struct ServerSessionAssign {
	SessionId sessionId; //!< Session Id assigned to player.
};

//! The game started.
//! TODO: This currently just wraps data. Extend to contain timestamps and other networking relevant info.
struct ServerGameStart {
	GameConfig config;
};

//! One accepted move, so the client can apply it to its position.
struct ServerGameDelta {
	GameDelta delta;
};

//! The game ended. Comes right after the delta of the last move.
struct ServerGameEnd {
	GameResult result;
};

struct ServerChat {
	Player player;       //!< Player who sent the message.
	unsigned messageId;  //!< Unique identifier.
	std::string message; //!< Chat message.
};


using ClientEvent = std::variant<ClientPutStone, ClientPass, ClientResign, ClientChat>;
using ServerEvent = std::variant<ServerSessionAssign, ServerGameStart, ServerGameDelta, ServerGameEnd, ServerChat>;

// Serialize typed events to JSON messages.
std::string toMessage(ClientEvent event);
std::string toMessage(ServerEvent event);

// Parse JSON messages into typed events. Returns empty on invalid input.
std::optional<ClientEvent> fromClientMessage(const std::string& message);
std::optional<ServerEvent> fromServerMessage(const std::string& message);

} // namespace tengen::network
