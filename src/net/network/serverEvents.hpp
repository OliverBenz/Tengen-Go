#pragma once

#include "network/core/protocol.hpp"
#include "network/types.hpp"

namespace tengen::network {

// Events for the server thread: from the network, and messages to send from any thread.
// Keep these small PODs so network callbacks remain cheap.
enum class ServerQueueEventType {
	ClientConnected,
	ClientDisconnected,
	ClientMessage,
	Send,
	Broadcast,
	Shutdown, //!< Last event the server thread handles. Queued by stop().
};

struct ServerQueueEvent {
	ServerQueueEventType type{};
	core::ConnectionId connectionId{core::ConnectionId::Empty}; //!< Network connection id.
	SessionId sessionId{SessionId::Empty};                      //!< Receiver of a Send.
	core::Message payload{};                                    //!< Message received, or to send. E.g. {"type":"put","x":3,"y":4}.
};

} // namespace tengen::network
