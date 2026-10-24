#pragma once

#include "network/core/protocol.hpp"

#include <cstdint>
#include <functional>
#include <memory>

namespace tengen {
namespace network::core {

//! Connection manager that runs an async accept loop on a dedicated IO thread.
//! \note    This is a thin wrapper: all heavy lifting is in Connection (async read/write).
//! \example Usage: construct with the callbacks, then start() once. Call stop() to shut down.
class TcpServer {
public:
	//! Run on the IO thread, and never before start().
	struct Callbacks {
		std::function<void(const ConnectionId&)> onConnect;
		std::function<void(const ConnectionId&, Message)> onMessage; //!< Gets the message to keep.
		std::function<void(const ConnectionId&)> onDisconnect;
	};

	explicit TcpServer(Callbacks callbacks, std::uint16_t port = DEFAULT_PORT);
	~TcpServer();

	TcpServer(const TcpServer&)            = delete;
	TcpServer& operator=(const TcpServer&) = delete;
	TcpServer(TcpServer&&)                 = delete;
	TcpServer& operator=(TcpServer&&)      = delete;

	bool start(); //!< Start accepting clients. Once a server stopped, it can't start again.
	void stop();  //!< Send what is queued, disconnect clients and stop the server. Waits up to a second for slow clients. Safe to call multiple times, not from a callback.

	bool send(ConnectionId connectionId, const Message& msg); //!< Send message to the client with given connectionId. Returns false if not found.
	void disconnect(ConnectionId connectionId);               //!< Close the connection once its queued messages are sent. onDisconnect follows.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide asio stuff in public interfaces.
};

} // namespace network::core
} // namespace tengen
