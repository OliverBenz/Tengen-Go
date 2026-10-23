#include "network/server.hpp"

#include "SafeQueue.hpp"
#include "network/core/tcpServer.hpp"
#include "serverEvents.hpp"
#include "sessionManager.hpp"

#include <cassert>
#include <mutex>
#include <thread>
#include <utility>

namespace tengen::network {

class Server::Implementation {
public:
	explicit Implementation(std::uint16_t port);

	void start();
	void stop();
	bool registerHandler(IServerHandler* handler);
	void setFirstSeat(Seat seat);

	bool send(SessionId sessionId, const ServerEvent& event); //!< Queue an event for one client. Any thread.
	bool broadcast(const ServerEvent& event);                 //!< Queue an event for all welcomed clients. Any thread.

private:
	void serverLoop();                                //!< Server thread: drain queue and act.
	void processEvent(const ServerQueueEvent& event); //!< Server loop calls this. Reads event type and distributes.

	Seat freeSeat() const;

	bool queueToSend(const ServerQueueEvent& event); //!< Queue a Send or Broadcast while running. False if it's empty or the server isn't running.

	// Server thread: the only one that reads the sessions.
	void sendNow(SessionId sessionId, const core::Message& message);
	void broadcastNow(const core::Message& message);

private:
	// Network callbacks (run on libNetwork threads) just enqueue events.
	void onClientConnected(core::ConnectionId connectionId);
	void onClientMessage(core::ConnectionId connectionId, const core::Message& payload);
	void onClientDisconnected(core::ConnectionId connectionId);

private:
	// Processing of server events.
	void processClientMessage(const ServerQueueEvent& event);         //!< Translate payload to network event and handle.
	void processClientConnect(const ServerQueueEvent& event);         //!< Creates session key.
	void processClientDisconnect(const ServerQueueEvent& event);      //!< Destroys session key.
	void processHello(SessionId sessionId, const ClientHello& hello); //!< Welcome the client with a seat, or reject it.

	// Hand a player's message to the handler.
	void handleNetworkEvent(Player player, const ClientHello& event);
	void handleNetworkEvent(Player player, const ClientPutStone& event);
	void handleNetworkEvent(Player player, const ClientPass& event);
	void handleNetworkEvent(Player player, const ClientResign& event);
	void handleNetworkEvent(Player player, const ClientChat& event);

private:
	enum class State : std::uint8_t {
		Created,
		Running,
		Stopped, //!< For good: a stopped server can't start again.
	};

	std::mutex m_lifecycleMutex; //!< start() and stop() run one at a time, so stop() always finds the threads start() made.
	std::mutex m_stateMutex;     //!< Guards the state, so nothing is queued behind the shutdown.
	State m_state{State::Created};
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
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	{
		std::lock_guard<std::mutex> stateLock(m_stateMutex);
		if (m_state != State::Created) {
			assert(m_state == State::Running && "A stopped server can't start again.");
			return;
		}
		m_state = State::Running;
	}

	// Network runs on its own IO thread; serverLoop drains the queue on its own thread.
	m_network.start();
	m_serverThread = std::thread([this] { serverLoop(); });
}

void Server::Implementation::stop() {
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	assert(std::this_thread::get_id() != m_serverThread.get_id()); // A handler can't stop its server.
	{
		// Under the state lock: what send() and broadcast() queued before goes out, nothing after.
		std::lock_guard<std::mutex> stateLock(m_stateMutex);
		const auto state = std::exchange(m_state, State::Stopped);
		if (state != State::Running) {
			return;
		}
		m_eventQueue.Push(ServerQueueEvent{.type = ServerQueueEventType::Shutdown});
	}

	m_serverThread.join(); // Handles every event before the shutdown, so the messages reach the connections.
	m_network.stop();      // The connections send them, then close.
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
	return queueToSend(ServerQueueEvent{.type = ServerQueueEventType::Send, .sessionId = sessionId, .payload = toMessage(event)});
}

bool Server::Implementation::broadcast(const ServerEvent& event) {
	return queueToSend(ServerQueueEvent{.type = ServerQueueEventType::Broadcast, .payload = toMessage(event)});
}

bool Server::Implementation::queueToSend(const ServerQueueEvent& event) {
	if (event.payload.empty()) {
		return false;
	}
	std::lock_guard<std::mutex> stateLock(m_stateMutex);
	if (m_state != State::Running) {
		return false;
	}
	m_eventQueue.Push(event);
	return true;
}

void Server::Implementation::sendNow(const SessionId sessionId, const core::Message& message) {
	if (const auto connectionId = m_sessionManager.getConnectionId(sessionId)) {
		m_network.send(connectionId, message);
	}
}

void Server::Implementation::broadcastNow(const core::Message& message) {
	m_sessionManager.forEachSession([&](const SessionContext& context) {
		if (context.isActive && context.seat != Seat::None) {
			m_network.send(context.connectionId, message);
		}
	});
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
	// stop() ends the loop with the shutdown event.
	for (auto event = m_eventQueue.Pop(); event.type != ServerQueueEventType::Shutdown; event = m_eventQueue.Pop()) {
		// The server thread must survive a throwing handler. The event is lost, the next one goes on.
		try {
			processEvent(event);
		} catch (...) {}
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
	case ServerQueueEventType::Send:
		sendNow(event.sessionId, event.payload);
		break;
	case ServerQueueEventType::Broadcast:
		broadcastNow(event.payload);
		break;
	case ServerQueueEventType::Shutdown:
		break; // serverLoop ends on it.
	}
}

void Server::Implementation::processClientConnect(const ServerQueueEvent& event) {
	// TODO: Possible to have this connectionId already registered? Yes, reconnect! Not thandled yet
	// The session starts without a seat. The client gets one with its hello.
	m_sessionManager.add(event.connectionId);
}

void Server::Implementation::processClientMessage(const ServerQueueEvent& event) {
	const auto sessionId = m_sessionManager.getSessionId(event.connectionId);
	if (!sessionId) {
		return;
	}

	// Server event message contains a network event. Parse and handle.
	const auto networkEvent = network::fromClientMessage(event.payload);
	if (!networkEvent) {
		return;
	}

	// A client has no seat until a 'hello', and the hello is the only message that counts.
	const auto seat = m_sessionManager.getSeat(sessionId);
	if (seat == Seat::None) {
		if (const auto* hello = std::get_if<ClientHello>(&*networkEvent)) {
			processHello(sessionId, *hello);
		}
		return;
	}

	const auto player = toPlayer(seat);
	if (!player) {
		return; // Non players don't get to do shit.
	}

	// Forward client intent to the game/app layer.
	if (m_handler) {
		std::visit([&](const auto& e) { handleNetworkEvent(*player, e); }, *networkEvent);
	}
}

void Server::Implementation::processHello(const SessionId sessionId, const ClientHello& hello) {
	if (hello.protocol != PROTOCOL_VERSION) {
		// Sent right away, so it goes out before the disconnect.
		sendNow(sessionId, toMessage(ServerReject{.reason = RejectReason::ProtocolMismatch}));
		m_network.disconnect(m_sessionManager.getConnectionId(sessionId));
		return;
	}

	const auto seat = freeSeat();
	m_sessionManager.setSeat(sessionId, seat);
	sendNow(sessionId, toMessage(ServerWelcome{.protocol = PROTOCOL_VERSION, .sessionId = sessionId, .seat = seat}));

	// Observers are no player; the handler only hears about players.
	const auto player = toPlayer(seat);
	if (m_handler && player) {
		m_handler->onPlayerJoined(*player);
	}
}

void Server::Implementation::handleNetworkEvent(const Player, const ClientHello&) {
	// The session is open already. A second hello changes nothing.
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
