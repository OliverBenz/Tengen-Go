#include "network/client.hpp"

#include "network/core/tcpClient.hpp"

#include <atomic>
#include <cstdint>

namespace tengen::network {

namespace {

DisconnectReason toReason(const core::DisconnectReason reason) {
	switch (reason) {
	case core::DisconnectReason::Requested:
		return DisconnectReason::ClosedLocally;
	case core::DisconnectReason::ConnectFailed:
		return DisconnectReason::ConnectFailed;
	case core::DisconnectReason::ConnectionLost:
		return DisconnectReason::ConnectionLost;
	}
	return DisconnectReason::ConnectionLost;
}

DisconnectReason toReason(const RejectReason reason) {
	switch (reason) {
	case RejectReason::ProtocolMismatch:
		return DisconnectReason::ProtocolMismatch;
	}
	return DisconnectReason::ProtocolError;
}

} // namespace

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
	std::atomic<SessionId> m_sessionId{0};                           //!< Set in callbacks, read by the owner.
	DisconnectReason m_closeReason{DisconnectReason::ClosedLocally}; //!< Hands the reason from close() to onDisconnect().

	core::TcpClient m_client; //!< Last, so it goes first and no callback reaches the members above.
};

Client::Implementation::Implementation()
    : m_client(core::TcpClient::Callbacks{
              .onMessage    = [this](const core::Message& message) { onMessage(message); },
              .onDisconnect = [this](const core::DisconnectReason reason) { onDisconnect(reason); },
      }) {
}

bool Client::Implementation::registerHandler(IClientHandler* handler) {
	if (m_handler) {
		return false;
	}
	m_handler = handler;
	return true;
}

bool Client::Implementation::connect(const std::string& host, std::uint16_t port) {
	// The server ignores everything before the hello.
	return m_client.connect(host, port, toMessage(ClientHello{.protocol = PROTOCOL_VERSION}));
}

void Client::Implementation::disconnect() {
	m_client.disconnect();
}

bool Client::Implementation::isConnected() const {
	return m_client.isConnected() && m_sessionId != 0;
}

bool Client::Implementation::send(const ClientEvent& event) {
	return m_client.send(toMessage(event));
}

SessionId Client::Implementation::sessionId() const {
	return m_sessionId;
}

void Client::Implementation::onMessage(const core::Message& message) {
	// A throwing handler must not take the IO thread down.
	try {
		const auto event = fromServerMessage(message);
		if (!event) {
			close(DisconnectReason::ProtocolError); // Same version on both sides, so the server is broken.
			return;
		}
		std::visit([&](const auto& e) { handleNetworkEvent(e); }, *event);
	} catch (...) { close(DisconnectReason::Error); }
}

void Client::Implementation::onDisconnect(const core::DisconnectReason reason) {
	const auto ourReason = reason == core::DisconnectReason::Requested ? m_closeReason : toReason(reason);
	if (m_handler) {
		// The connection is over, but the IO thread must survive a throwing handler.
		try {
			m_handler->onDisconnected(ourReason);
		} catch (...) {}
	}
	// After the handler, which may still want the id. A new welcome can't come earlier: callbacks run one at a time.
	m_sessionId = 0;
}

void Client::Implementation::close(const DisconnectReason reason) {
	// Inside a callback, disconnect() reports right away, so the reason is set for onDisconnect().
	m_closeReason = reason;
	m_client.disconnect();
	m_closeReason = DisconnectReason::ClosedLocally;
}

void Client::Implementation::handleNetworkEvent(const ServerWelcome& event) {
	// The server rejects other versions, and 0 means no session. Either way, the server is broken.
	if (event.protocol != PROTOCOL_VERSION || event.sessionId == 0) {
		close(DisconnectReason::ProtocolError);
		return;
	}

	m_sessionId = event.sessionId; // Before the handler, so isConnected() is true in onWelcome.
	if (m_handler) {
		m_handler->onWelcome(toPlayer(event.seat));
	}
}

void Client::Implementation::handleNetworkEvent(const ServerReject& event) {
	close(toReason(event.reason));
}

void Client::Implementation::handleNetworkEvent(const ServerGameStart& event) {
	if (m_handler) {
		m_handler->onGameStart(event.config);
	}
}

void Client::Implementation::handleNetworkEvent(const ServerGameDelta& event) {
	if (m_handler) {
		m_handler->onGameDelta(event.delta);
	}
}

void Client::Implementation::handleNetworkEvent(const ServerGameEnd& event) {
	if (m_handler) {
		m_handler->onGameEnd(event.result);
	}
}

void Client::Implementation::handleNetworkEvent(const ServerChat& event) {
	if (m_handler) {
		m_handler->onChatMessage(event.player, event.messageId, event.message);
	}
}


Client::Client()
    : m_pimpl(std::make_unique<Implementation>()) {
}

Client::~Client() = default;

bool Client::registerHandler(IClientHandler* handler) {
	return m_pimpl->registerHandler(handler);
}

bool Client::connect(const std::string& host) {
	return m_pimpl->connect(host, core::DEFAULT_PORT);
}

bool Client::connect(const std::string& host, std::uint16_t port) {
	return m_pimpl->connect(host, port);
}

void Client::disconnect() {
	m_pimpl->disconnect();
}

bool Client::isConnected() const {
	return m_pimpl->isConnected();
}

bool Client::send(const ClientEvent& event) {
	return m_pimpl->send(event);
}

SessionId Client::sessionId() const {
	return m_pimpl->sessionId();
}

} // namespace tengen::network
