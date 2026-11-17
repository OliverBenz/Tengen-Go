#include "serverImpl.hpp"

#include <cassert>
#include <mutex>
#include <thread>
#include <utility>
#include <variant>

namespace tengen::network {

namespace {
//! Combines lambdas into one visitor for std::visit.
template <class... Ts>
struct Overloaded : Ts... {
	using Ts::operator()...;
};

//! Connection::send would drop an empty or oversized message silently; send() and broadcast() refuse it so the caller knows.
bool isSendable(const core::Message& payload) {
	return !payload.empty() && payload.size() <= core::MAX_PAYLOAD_BYTES;
}
} // namespace

// Network callbacks are kept thin: they only enqueue events. None runs before start(), so capturing this is safe here.
Server::Implementation::Implementation(std::uint16_t port)
    : m_network(
              core::TcpServer::Callbacks{
                      .onConnect    = [this](core::ConnectionId connectionId) { onClientConnected(connectionId); },
                      .onMessage    = [this](core::ConnectionId connectionId, core::Message payload) { onClientMessage(connectionId, std::move(payload)); },
                      .onDisconnect = [this](core::ConnectionId connectionId) { onClientDisconnected(connectionId); },
              },
              port) {
}

bool Server::Implementation::start() {
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	{
		std::lock_guard<std::mutex> stateLock(m_stateMutex);
		if (m_state != State::Created) {
			assert(m_state == State::Running && "A stopped server can't start again.");
			return m_state == State::Running;
		}
		m_state = State::Running; // Before the network starts, so the first client's events are queued.
	}

	// Network runs on its own IO thread; serverLoop drains the queue on its own thread.
	if (!m_network.start()) {
		// No port, no clients: stopped for good, like after stop(). Nothing was queued that needs handling.
		std::lock_guard<std::mutex> stateLock(m_stateMutex);
		m_state = State::Stopped;
		return false;
	}
	m_serverThread = std::thread([this] { serverLoop(); });
	return true;
}

void Server::Implementation::stop() {
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	if (std::this_thread::get_id() == m_serverThread.get_id()) {
		assert(false && "A handler can't stop its server.");
		return;
	}

	{
		// Under the state lock: what send() and broadcast() queued before goes out, nothing after.
		std::lock_guard<std::mutex> stateLock(m_stateMutex);
		const auto state = std::exchange(m_state, State::Stopped);
		if (state != State::Running) {
			return;
		}
		m_eventQueue.Push(Shutdown{});
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
	auto payload = toMessage(event);
	if (!isSendable(payload)) {
		return false;
	}
	return queue(Send{.sessionId = sessionId, .payload = std::move(payload)});
}

bool Server::Implementation::broadcast(const ServerEvent& event) {
	auto payload = toMessage(event);
	if (!isSendable(payload)) {
		return false;
	}
	return queue(Broadcast{.payload = std::move(payload)});
}

bool Server::Implementation::queue(ServerQueueEvent event) {
	std::lock_guard<std::mutex> stateLock(m_stateMutex);
	if (m_state != State::Running) {
		return false;
	}
	m_eventQueue.Push(std::move(event));
	return true;
}

void Server::Implementation::sendNow(const SessionId sessionId, const core::Message& message) {
	if (const auto connectionId = m_sessionManager.getConnectionId(sessionId); connectionId != core::ConnectionId::Empty) {
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
	queue(ClientConnected{.connectionId = connectionId});
}

void Server::Implementation::onClientMessage(core::ConnectionId connectionId, core::Message payload) {
	queue(ClientMessage{.connectionId = connectionId, .payload = std::move(payload)});
}

void Server::Implementation::onClientDisconnected(core::ConnectionId connectionId) {
	queue(ClientDisconnected{.connectionId = connectionId});
}

void Server::Implementation::serverLoop() {
	while (true) {
		const auto event = m_eventQueue.Pop();
		if (std::holds_alternative<Shutdown>(event)) {
			return; // Queued last by stop().
		}

		// The server thread must survive a throwing handler. The event is lost, the next one goes on.
		try {
			std::visit([this](const auto& e) { processEvent(e); }, event);
		} catch (...) {} // NOLINT(bugprone-empty-catch)
	}
}

void Server::Implementation::processEvent(const Send& event) {
	sendNow(event.sessionId, event.payload);
}

void Server::Implementation::processEvent(const Broadcast& event) {
	broadcastNow(event.payload);
}

void Server::Implementation::processEvent(const Shutdown&) {
	// serverLoop ends on it.
}

void Server::Implementation::processEvent(const ClientConnected& event) {
	// TODO: Possible to have this connectionId already registered? Yes, reconnect! Not thandled yet
	// The session starts without a seat. The client gets one with its hello.
	m_sessionManager.add(event.connectionId);
}

void Server::Implementation::processEvent(const ClientMessage& event) {
	const auto sessionId = m_sessionManager.getSessionId(event.connectionId);
	if (sessionId == SessionId::Empty) {
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
		std::visit(Overloaded{
		                   [](const ClientHello&) {}, // The session is open already. A second hello changes nothing.
		                   [&](const ClientPutStone& e) { m_handler->onPlace(*player, e.c); },
		                   [&](const ClientPass&) { m_handler->onPass(*player); },
		                   [&](const ClientResign&) { m_handler->onResign(*player); },
		                   [&](const ClientChat& e) { m_handler->onChat(*player, e.message); },
		           },
		           *networkEvent);
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

void Server::Implementation::processEvent(const ClientDisconnected& event) {
	const auto sessionId = m_sessionManager.getSessionId(event.connectionId);
	if (sessionId == SessionId::Empty) {
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

	if (m_sessionManager.getConnectionIdBySeat(m_firstSeat) == core::ConnectionId::Empty) {
		return m_firstSeat;
	}
	if (m_sessionManager.getConnectionIdBySeat(secondSeat) == core::ConnectionId::Empty) {
		return secondSeat;
	}
	return Seat::Observer;
}

} // namespace tengen::network
