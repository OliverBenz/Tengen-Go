#pragma once

#include "network/client.hpp"

#include "network/core/tcpClient.hpp"

#include <atomic>
#include <cstdint>
#include <string>

namespace tengen::network {

class Client::Implementation {
public:
	Implementation();

	bool registerHandler(IClientHandler* handler);

	bool connect(const std::string& host, std::uint16_t port);
	void disconnect();
	bool isConnected() const;

	bool send(const ClientEvent& event);

	SessionId sessionId() const;

private:
	// TcpClient callbacks, one at a time.
	void onConnecting();
	void onMessage(const core::Message& message);
	void onDisconnect(core::DisconnectReason reason); //!< The only place that tells the handler a connection ended.

	void close(DisconnectReason reason); //!< End the connection from inside a callback.

private:
	// Unwrap the messages, so the handler only sees model data.
	void handleNetworkEvent(const ServerWelcome& event);
	void handleNetworkEvent(const ServerReject& event);
	void handleNetworkEvent(const ServerGameStart& event);
	void handleNetworkEvent(const ServerGameDelta& event);
	void handleNetworkEvent(const ServerGameEnd& event);
	void handleNetworkEvent(const ServerChat& event);

private:
	IClientHandler* m_handler{nullptr};
	std::atomic<SessionId> m_sessionId{SessionId::Empty};            //!< Set in callbacks, read by the owner.
	DisconnectReason m_closeReason{DisconnectReason::ClosedLocally}; //!< Hands the reason from close() to onDisconnect().

	core::TcpClient m_client; //!< Last, so it goes first and no callback reaches the members above.
};

} // namespace tengen::network
