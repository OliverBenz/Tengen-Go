#pragma once

#include "network/core/protocol.hpp"
#include "network/types.hpp"

#include <variant>

namespace tengen::network {

// Events for the server thread: from the network, and messages to send from any thread.
// Keep these small so network callbacks remain cheap.
struct ClientConnected {
	core::ConnectionId connectionId; //!< Network connection id.
};

struct ClientDisconnected {
	core::ConnectionId connectionId; //!< Network connection id.
};

struct ClientMessage {
	core::ConnectionId connectionId; //!< Network connection id.
	core::Message payload;           //!< Message received. E.g. {"type":"put","x":3,"y":4}.
};

struct Send {
	SessionId sessionId;   //!< Receiver.
	core::Message payload; //!< Message to send.
};

struct Broadcast {
	core::Message payload; //!< Message to send to all welcomed clients.
};

struct Shutdown {}; //!< Last event the server thread handles. Queued by stop().

using ServerQueueEvent = std::variant<ClientConnected, ClientDisconnected, ClientMessage, Send, Broadcast, Shutdown>;

} // namespace tengen::network
