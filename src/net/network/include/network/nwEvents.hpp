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

//! First message on a connection. The server ignores everything else until it arrives.
struct ClientHello {
	ProtocolVersion protocol; //!< Protocol version of the client.
};
struct ClientPutStone {
	Coord c;
};
struct ClientPass {};
struct ClientResign {};
struct ClientChat {
	std::string message;
};

// Server Events (server -> client)
//! Answer to a hello with a matching version. Opens the session.
struct ServerWelcome {
	ProtocolVersion protocol; //!< Protocol version of the server.
	SessionId sessionId;      //!< Session Id assigned to the client.
	Seat seat;                //!< Seat assigned to the client. Never Seat::None.
};

//! Answer to a refused hello. The server closes the connection right after.
struct ServerReject {
	RejectReason reason;
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


using ClientEvent = std::variant<ClientHello, ClientPutStone, ClientPass, ClientResign, ClientChat>;
using ServerEvent = std::variant<ServerWelcome, ServerReject, ServerGameStart, ServerGameDelta, ServerGameEnd, ServerChat>;

// Serialize typed events to JSON messages.
std::string toMessage(const ClientEvent& event);
std::string toMessage(const ServerEvent& event);

// Parse JSON messages into typed events. Returns empty on invalid input.
std::optional<ClientEvent> fromClientMessage(const std::string& message);
std::optional<ServerEvent> fromServerMessage(const std::string& message);

} // namespace tengen::network
