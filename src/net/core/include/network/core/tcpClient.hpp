#pragma once

#include "network/core/protocol.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace tengen {
namespace network::core {

//! Why a connection ended.
enum class DisconnectReason : std::uint8_t {
	Requested,      //!< disconnect() was called.
	ConnectFailed,  //!< The server could not be reached.
	ConnectionLost, //!< The server closed the connection, or it broke.
};

//! TCP client with its own IO thread, like TcpServer.
//! No call waits on the network, and all are safe from any thread. connect() and disconnect() wait for a running callback.
//! Callbacks run one at a time on the IO thread; only onDisconnect(Requested) runs on the thread calling disconnect().
//! Never destroy the client from a callback.
class TcpClient {
public:
	struct Callbacks {
		std::function<void(const Message&)> onMessage;
		std::function<void(DisconnectReason)> onDisconnect; //!< Once per successful connect(). Not when the client is destroyed.
	};

	explicit TcpClient(Callbacks callbacks);
	~TcpClient();

	TcpClient(const TcpClient&)            = delete;
	TcpClient& operator=(const TcpClient&) = delete;
	TcpClient(TcpClient&&)                 = delete;
	TcpClient& operator=(TcpClient&&)      = delete;

	//! Start connecting. False if already connected or connecting, or if the first message is too big.
	//! The first message goes out before anything you send().
	bool connect(std::string host, std::uint16_t port = DEFAULT_PORT, std::optional<Message> firstMessage = std::nullopt);

	bool send(const Message& message); //!< Sends, or queues until connected. False if not connected or connecting, or if too big.
	bool disconnect();                 //!< Calls onDisconnect(Requested) before it returns. False if there was nothing to end.
	bool isConnected() const;          //!< False while still connecting.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide asio stuff in public interfaces.
};

} // namespace network::core
} // namespace tengen
