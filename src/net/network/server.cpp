#include "network/server.hpp"

#include "SafeQueue.hpp"
#include "network/core/tcpServer.hpp"
#include "serverEvents.hpp"
#include "sessionManager.hpp"

#include <atomic>
#include <cassert>
#include <thread>

namespace tengen::network {

class Server::Implementation {
public:
	explicit Implementation(std::uint16_t port);

	void start();
	void stop();
	bool registerHandler(IServerHandler* handler);
	void setFirstSeat(Seat seat);

	bool send(SessionId sessionId, const ServerEvent& event); //!< Send event to client with given sessionId.
	bool broadcast(const ServerEvent& event);                 //!< Send event to all connected clients.

private:
	void serverLoop();                                //!< Server thread: drain queue and act.
	void processEvent(const ServerQueueEvent& event); //!< Server loop calls this. Reads event type and distributes.

	Seat freeSeat() const;

private:
	// Network callbacks (run on libNetwork threads) just enqueue events.
	void onClientConnected(core::ConnectionId connectionId);
	void onClientMessage(core::ConnectionId connectionId, const core::Message& payload);
	void onClientDisconnected(core::ConnectionId connectionId);

private:
	// Processing of server events.
	void processClientMessage(const ServerQueueEvent& event);    //!< Translate payload to network event and handle.
	void processClientConnect(const ServerQueueEvent& event);    //!< Creates session key.
	void processClientDisconnect(const ServerQueueEvent& event); //!< Destroys session key.
	void processShutdown(const ServerQueueEvent& event);         //!< Shutdown server.

	// Hand a player's message to the handler.
	void handleNetworkEvent(Player player, const ClientPutStone& event);
	void handleNetworkEvent(Player player, const ClientPass& event);
	void handleNetworkEvent(Player player, const ClientResign& event);
	void handleNetworkEvent(Player player, const ClientChat& event);

private:
	std::atomic<bool> m_isRunning{false};
	std::thread m_serverThread;

	SessionManager m_sessionManager;
	core::TcpServer m_network;

	IServerHandler* m_handler{nullptr};       //!< The class that will handle server events.
	Seat m_firstSeat{Seat::Black};            //!< Seat handed out first while both are free.
	SafeQueue<ServerQueueEvent> m_eventQueue; //!< Event queue between network threads and server thread.
};

Server::Implementation::Implementation(std::uint16_t port) : m_network{port} {
	// Wire up network callbacks but keep them thin: they only enqueue events.
	core::TcpServer::Callbacks callbacks;
	callbacks.onConnect    = [this](core::ConnectionId connectionId) { return onClientConnected(connectionId); };
	callbacks.onMessage    = [this](core::ConnectionId connectionId, const core::Message& payload) { return onClientMessage(connectionId, payload); };
	callbacks.onDisconnect = [this](core::ConnectionId connectionId) { onClientDisconnected(connectionId); };
	m_network.connect(callbacks);
}

void Server::Implementation::start() {
	if (m_isRunning.exchange(true)) {
		return;
	}

	// Network runs on its own IO thread; serverLoop drains the queue on its own thread.
	m_network.start();
	m_serverThread = std::thread([this] { serverLoop(); });
}

void Server::Implementation::stop() {
	// Wake serverLoop and stop network.
	if (m_isRunning.exchange(false)) {
		m_eventQueue.Push(ServerQueueEvent{.type = ServerQueueEventType::Shutdown});
	}
	m_network.stop();

	try {
		m_eventQueue.Release();
	} catch (...) {
		// Release never throws, but keep intent explicit.
	}

	if (m_serverThread.joinable()) {
		m_serverThread.join();
	}
}

bool Server::Implementation::registerHandler(IServerHandler* handler) {
	if (m_handler) {
		return false;
	}
	m_handler = handler;
	return true;
}

void Server::Implementation::setFirstSeat(const Seat seat) {
	assert(isPlayer(seat));
	m_firstSeat = seat;
}

bool Server::Implementation::send(SessionId sessionId, const ServerEvent& event) {
	const auto connectionId = m_sessionManager.getConnectionId(sessionId);
	if (!connectionId) {
		return false;
	}
	const auto message = toMessage(event);
	if (message.empty()) {
		return false;
	}
	return m_network.send(connectionId, message);
}

bool Server::Implementation::broadcast(const ServerEvent& event) {
	const auto message = toMessage(event);
	if (message.empty()) {
		return false;
	}
	bool anySent = false;

	m_sessionManager.forEachSession([&](const SessionContext& context) {
		if (!context.isActive || context.seat == Seat::None) {
			return;
		}
		if (m_network.send(context.connectionId, message)) {
			anySent = true;
		}
	});

	return anySent;
}

void Server::Implementation::onClientConnected(core::ConnectionId connectionId) {
	m_eventQueue.Push(ServerQueueEvent{.type = ServerQueueEventType::ClientConnected, .connectionId = connectionId});
}

void Server::Implementation::onClientMessage(core::ConnectionId connectionId, const core::Message& payload) {
	m_eventQueue.Push(ServerQueueEvent{
	        .type         = ServerQueueEventType::ClientMessage,
	        .connectionId = connectionId,
	        .payload      = payload,
	});
}

void Server::Implementation::onClientDisconnected(core::ConnectionId connectionId) {
	m_eventQueue.Push(ServerQueueEvent{.type = ServerQueueEventType::ClientDisconnected, .connectionId = connectionId});
}

void Server::Implementation::serverLoop() {
	while (m_isRunning) {
		try {
			const auto event = m_eventQueue.Pop();
			processEvent(event);
		} catch (const std::exception&) {
			if (!m_isRunning) {
				break;
			}
		}
	}
}

void Server::Implementation::processEvent(const ServerQueueEvent& event) {
	switch (event.type) {
	case ServerQueueEventType::ClientConnected:
		processClientConnect(event);
		break;
	case ServerQueueEventType::ClientDisconnected:
		processClientDisconnect(event);
		break;
	case ServerQueueEventType::ClientMessage:
		processClientMessage(event);
		break;
	case ServerQueueEventType::Shutdown:
		processShutdown(event);
		break;
	}
}

void Server::Implementation::processClientConnect(const ServerQueueEvent& event) {
	// TODO: Possible to have this connectionId already registered? Yes, reconnect! Not thandled yet
	const auto sessionId = m_sessionManager.add(event.connectionId);
	const auto seat      = freeSeat();

	// Store sessionId & send to client
	m_sessionManager.setSeat(sessionId, seat);
	send(sessionId, ServerSessionAssign{.sessionId = sessionId});

	// Observers are no player; the handler only hears about players.
	const auto player = toPlayer(seat);
	if (m_handler && player) {
		m_handler->onPlayerJoined(*player);
	}
}

void Server::Implementation::processClientMessage(const ServerQueueEvent& event) {
	const auto sessionId = m_sessionManager.getSessionId(event.connectionId);
	if (!sessionId) {
		return;
	}

	const auto player = toPlayer(m_sessionManager.getSeat(sessionId));
	if (!player) {
		return; // Non players don't get to do shit.
	}

	// Server event message contains a network event. Parse and handle.
	const auto networkEvent = network::fromClientMessage(event.payload);
	if (!networkEvent) {
		return;
	}

	// Forward client intent to the game/app layer.
	if (m_handler) {
		std::visit([&](const auto& e) { handleNetworkEvent(*player, e); }, *networkEvent);
	}
}

void Server::Implementation::handleNetworkEvent(const Player player, const ClientPutStone& event) {
	m_handler->onPlace(player, event.c);
}

void Server::Implementation::handleNetworkEvent(const Player player, const ClientPass&) {
	m_handler->onPass(player);
}

void Server::Implementation::handleNetworkEvent(const Player player, const ClientResign&) {
	m_handler->onResign(player);
}

void Server::Implementation::handleNetworkEvent(const Player player, const ClientChat& event) {
	m_handler->onChat(player, event.message);
}

void Server::Implementation::processClientDisconnect(const ServerQueueEvent& event) {
	const auto sessionId = m_sessionManager.getSessionId(event.connectionId);
	if (!sessionId) {
		return; // Should never happen
	}

	// Remove session
	const auto player = toPlayer(m_sessionManager.getSeat(sessionId));
	m_sessionManager.setDisconnected(sessionId);

	if (m_handler && player) {
		m_handler->onPlayerLeft(*player); // Server might want to pause timer.
	}
}

void Server::Implementation::processShutdown(const ServerQueueEvent&) {
	m_isRunning = false;
}

Seat Server::Implementation::freeSeat() const {
	const Seat secondSeat = m_firstSeat == Seat::Black ? Seat::White : Seat::Black;

	if (!m_sessionManager.getConnectionIdBySeat(m_firstSeat)) {
		return m_firstSeat;
	}
	if (!m_sessionManager.getConnectionIdBySeat(secondSeat)) {
		return secondSeat;
	}
	return Seat::Observer;
}


Server::Server() : m_pimpl(std::make_unique<Implementation>(core::DEFAULT_PORT)) {
}

Server::Server(std::uint16_t port) : m_pimpl(std::make_unique<Implementation>(port)) {
}

Server::~Server() {
	stop();
}

void Server::start() {
	m_pimpl->start();
}

void Server::stop() {
	m_pimpl->stop();
}

bool Server::registerHandler(IServerHandler* handler) {
	return m_pimpl->registerHandler(handler);
}

void Server::setFirstSeat(const Seat seat) {
	m_pimpl->setFirstSeat(seat);
}

bool Server::send(SessionId sessionId, const ServerEvent& event) {
	return m_pimpl->send(sessionId, event);
}

bool Server::broadcast(const ServerEvent& event) {
	return m_pimpl->broadcast(event);
}

} // namespace tengen::network
