#include "network/core/tcpServer.hpp"
#include "connection.hpp"

#include <asio.hpp>
#include <asio/ip/tcp.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

namespace tengen {
namespace network::core {

static constexpr auto CLOSE_TIMEOUT = std::chrono::seconds(1); //!< How long stop() lets clients take the messages still queued for them.

class TcpServer::Implementation {
public:
	Implementation(std::uint16_t port);

	void start();
	void connect(Callbacks callbacks);
	void stop();

	bool send(ConnectionId connectionId, const Message& msg);
	void disconnect(ConnectionId connectionId);

private:
	enum class State : std::uint8_t {
		Created,
		Running,
		Stopped, //!< For good: a stopped server can't start again.
	};

	void doAccept();                                                                //!< Start async accept loop.
	bool createConnection(asio::ip::tcp::socket socket, ConnectionId connectionId); //!< Create and add new connection to map.

private:
	asio::io_context m_ioContext{};
	asio::ip::tcp::acceptor m_acceptor;
	std::optional<asio::executor_work_guard<asio::io_context::executor_type>> m_workGuard;
	bool m_acceptorReady{false};

	std::thread m_ioThread;                     //!< IO context thread.
	std::atomic<State> m_state{State::Created}; //!< Read on the IO thread.
	std::mutex m_lifecycleMutex;                //!< start() and stop() run one at a time, so stop() always finds the thread start() made. Also makes a second stop() wait for the first.

	Callbacks m_callbacks; //!< Callback functions to signal events.

	ConnectionId m_nextConnectionId{1u};                                         //!< Only used on the IO thread.
	std::unordered_map<ConnectionId, std::shared_ptr<Connection>> m_connections; //!< Active connections.
	std::mutex m_connectionsMutex;                                               //!< Handle concurrency.
	std::condition_variable m_connectionsGone;                                   //!< The last connection left the map. stop() waits for it.
};


TcpServer::Implementation::Implementation(std::uint16_t port) : m_acceptor(m_ioContext) {
	// Do a manual open/bind/listen so we can stay in error_code land and avoid throws.
	asio::error_code ec;
	m_acceptor.open(asio::ip::tcp::v4(), ec);
	if (ec) {
		return;
	}
	m_acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true), ec);
	if (ec) {
		return;
	}
	m_acceptor.bind(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port), ec);
	if (ec) {
		return;
	}
	m_acceptor.listen(asio::socket_base::max_listen_connections, ec);
	if (ec) {
		return;
	}
	m_acceptorReady = true;
}

void TcpServer::Implementation::start() {
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	auto expected = State::Created;
	if (!m_state.compare_exchange_strong(expected, State::Running)) {
		assert(expected == State::Running && "A stopped server can't start again.");
		return;
	}
	// If acceptor init failed during construction, don't start a dead server.
	if (!m_acceptorReady) {
		m_state = State::Stopped;
		return;
	}

	m_workGuard.emplace(asio::make_work_guard(m_ioContext));
	doAccept();
	m_ioThread = std::thread([this]() { m_ioContext.run(); });
}

void TcpServer::Implementation::connect(Callbacks callbacks) {
	m_callbacks = std::move(callbacks);
}

void TcpServer::Implementation::stop() {
	std::lock_guard<std::mutex> lifecycleLock(m_lifecycleMutex);
	if (m_state.exchange(State::Stopped) != State::Running) {
		return;
	}
	assert(std::this_thread::get_id() != m_ioThread.get_id()); // A callback can't stop its server.

	// Each connection sends what is queued, then closes and leaves the map. New clients are turned away by the state.
	{
		std::unique_lock<std::mutex> lock(m_connectionsMutex);
		for (auto& [id, connection]: m_connections) {
			connection->close();
		}
		m_connectionsGone.wait_for(lock, CLOSE_TIMEOUT, [this] { return m_connections.empty(); });
	}

	// Stop the IO thread before touching the acceptor and sockets: its handlers use them, and they aren't thread-safe.
	m_workGuard.reset();
	m_ioContext.stop();
	m_ioThread.join();

	asio::error_code ec;
	m_acceptor.close(ec);

	// Clients that didn't take their messages in time are cut off.
	std::unordered_map<ConnectionId, std::shared_ptr<Connection>> connections;
	{
		std::lock_guard<std::mutex> lock(m_connectionsMutex);
		connections.swap(m_connections);
	}
	for (auto& [id, conn]: connections) {
		conn->stop();
	}
}

bool TcpServer::Implementation::send(ConnectionId connectionId, const Message& msg) {
	std::lock_guard<std::mutex> lock(m_connectionsMutex);

	if (m_connections.contains(connectionId)) {
		m_connections.at(connectionId)->send(msg);
		return true;
	}

	return false;
}

void TcpServer::Implementation::disconnect(ConnectionId connectionId) {
	std::lock_guard<std::mutex> lock(m_connectionsMutex);

	// The connection erases itself from the map in onDisconnect.
	if (m_connections.contains(connectionId)) {
		m_connections.at(connectionId)->close();
	}
}

void TcpServer::Implementation::doAccept() {
	m_acceptor.async_accept([this](asio::error_code ec, asio::ip::tcp::socket socket) {
		if (!ec) {
			// The state is checked under the lock: else stop() could close the connections before this one is added.
			std::lock_guard<std::mutex> lock(m_connectionsMutex);
			if (m_state != State::Running) {
				return; // Turned away: the socket closes here.
			}
			const auto connectionId = m_nextConnectionId++;
			if (createConnection(std::move(socket), connectionId)) {
				assert(m_connections.contains(connectionId));
				m_connections.at(connectionId)->start();
			}
		}

		if (m_state == State::Running) {
			doAccept();
		}
	});
}

bool TcpServer::Implementation::createConnection(asio::ip::tcp::socket socket, ConnectionId connectionId) {
	if (m_connections.contains(connectionId)) {
		return false;
	}

	Connection::Callbacks callbacks;
	callbacks.onConnect = [this](Connection& connection) {
		if (m_callbacks.onConnect) {
			m_callbacks.onConnect(connection.connectionId());
		}
	};
	callbacks.onMessage = [this](Connection& connection, Message message) {
		if (m_callbacks.onMessage) {
			m_callbacks.onMessage(connection.connectionId(), std::move(message));
		}
	};
	callbacks.onDisconnect = [this](Connection& connection) {
		const auto index = connection.connectionId();
		{
			std::lock_guard<std::mutex> lock(m_connectionsMutex);
			m_connections.erase(index);
			if (m_connections.empty()) {
				m_connectionsGone.notify_all();
			}
		}
		if (m_callbacks.onDisconnect) {
			m_callbacks.onDisconnect(index);
		}
	};

	try {
		auto connection           = std::make_shared<Connection>(std::move(socket), connectionId, std::move(callbacks));
		const auto [it, inserted] = m_connections.try_emplace(connectionId, std::move(connection));
		return inserted;
	} catch (...) { return false; }
}


TcpServer::TcpServer(std::uint16_t port) : m_pimpl(std::make_unique<Implementation>(port)) {
}

TcpServer::~TcpServer() {
	stop();
}

void TcpServer::start() {
	m_pimpl->start();
}

void TcpServer::connect(Callbacks callbacks) {
	m_pimpl->connect(callbacks);
}

void TcpServer::stop() {
	m_pimpl->stop();
}

bool TcpServer::send(ConnectionId connectionId, const Message& msg) {
	return m_pimpl->send(connectionId, msg);
}

void TcpServer::disconnect(ConnectionId connectionId) {
	m_pimpl->disconnect(connectionId);
}


} // namespace network::core
} // namespace tengen
