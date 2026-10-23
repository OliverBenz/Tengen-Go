#include "network/core/tcpClient.hpp"
#include "connection.hpp"

#include <asio.hpp>

#include <cassert>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace tengen {
namespace network::core {

static constexpr ConnectionId CONNECTION_ID = 1u; //!< One connection at a time, so the id doesn't matter.

class TcpClient::Implementation {
public:
	explicit Implementation(Callbacks callbacks);
	~Implementation();

	bool connect(std::string host, std::uint16_t port, std::optional<Message> firstMessage);
	bool disconnect();
	bool isConnected() const;
	bool send(const Message& message);

private:
	enum class State : std::uint8_t {
		Idle,
		Connecting,
		Connected,
	};

	// IO thread. An attempt is current while its socket is m_socket.
	void onConnected(const std::shared_ptr<asio::ip::tcp::socket>& socket);
	void onConnectFailed(const std::shared_ptr<asio::ip::tcp::socket>& socket);
	void onConnectionMessage(const Connection& connection, const Message& message);
	void onConnectionEnded(const Connection& connection);

	bool release(); //!< Drop the connection or attempt without reporting it. Needs m_callbackMutex.

	bool isCurrent(const Connection& connection) const; //!< False once the client let go of it.
	bool isIoThread() const;

private:
	asio::io_context m_ioContext{};
	asio::executor_work_guard<asio::io_context::executor_type> m_workGuard; //!< Keeps run() going while idle.
	std::thread m_ioThread;

	Callbacks m_callbacks;
	std::recursive_mutex m_callbackMutex; //!< Held during callbacks. Recursive, so callbacks can connect() and disconnect().

	mutable std::mutex m_mutex; //!< Guards the state below. Never held during callbacks.
	State m_state{State::Idle};
	std::shared_ptr<asio::ip::tcp::resolver> m_resolver; //!< While connecting.
	std::shared_ptr<asio::ip::tcp::socket> m_socket;     //!< While connecting. Marks the current attempt.
	std::shared_ptr<Connection> m_connection;            //!< While connected.
	std::vector<Message> m_pending;                      //!< Queued while connecting.
};

TcpClient::Implementation::Implementation(Callbacks callbacks) : m_workGuard(asio::make_work_guard(m_ioContext)), m_callbacks(std::move(callbacks)) {
	m_ioThread = std::thread([this] { m_ioContext.run(); });
}

TcpClient::Implementation::~Implementation() {
	assert(!isIoThread()); // A callback can't destroy its client.

	{
		std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
		release(); // No report: the owner is going away.
	}
	m_workGuard.reset();
	m_ioContext.stop(); // Drops anything still queued.
	m_ioThread.join();
}

bool TcpClient::Implementation::connect(std::string host, const std::uint16_t port, std::optional<Message> firstMessage) {
	if (firstMessage && firstMessage->size() > MAX_PAYLOAD_BYTES) {
		return false;
	}

	// After a running callback, so the last connection's end is reported first.
	std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
	std::lock_guard<std::mutex> lock(m_mutex);
	if (m_state != State::Idle) {
		return false;
	}

	m_state    = State::Connecting;
	m_resolver = std::make_shared<asio::ip::tcp::resolver>(m_ioContext);
	m_socket   = std::make_shared<asio::ip::tcp::socket>(m_ioContext);
	if (firstMessage) {
		m_pending.push_back(std::move(*firstMessage)); // Same lock as the state, so nothing gets ahead of it.
	}

	// From here on, the resolver and socket belong to the IO thread. The handlers keep them alive.
	asio::post(m_ioContext, [this, resolver = m_resolver, socket = m_socket, host = std::move(host), port] {
		resolver->async_resolve(host, std::to_string(port), [this, resolver, socket](const asio::error_code& ec, const asio::ip::tcp::resolver::results_type& endpoints) {
			if (ec) {
				onConnectFailed(socket);
				return;
			}
			asio::async_connect(*socket, endpoints, [this, socket](const asio::error_code& connectEc, const asio::ip::tcp::endpoint&) {
				if (connectEc) {
					onConnectFailed(socket);
					return;
				}
				onConnected(socket);
			});
		});
	});
	return true;
}

bool TcpClient::Implementation::disconnect() {
	// After a running callback, and no other until the end is reported.
	std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
	if (!release()) {
		return false;
	}

	if (m_callbacks.onDisconnect) {
		m_callbacks.onDisconnect(DisconnectReason::Requested);
	}
	return true;
}

bool TcpClient::Implementation::release() {
	std::lock_guard<std::mutex> lock(m_mutex);
	if (m_state == State::Idle) {
		return false;
	}

	if (m_connection) {
		m_connection->close(); // Sends what is queued, then closes.
		m_connection.reset();
	}

	// asio objects aren't thread-safe, so cancel on the IO thread. Clearing the members outdates the attempt's handlers.
	if (m_resolver) {
		asio::post(m_ioContext, [resolver = std::exchange(m_resolver, nullptr), socket = std::exchange(m_socket, nullptr)] {
			resolver->cancel();
			asio::error_code ec;
			socket->close(ec);
		});
	}
	m_pending.clear();
	m_state = State::Idle;
	return true;
}

bool TcpClient::Implementation::isConnected() const {
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_state == State::Connected;
}

bool TcpClient::Implementation::send(const Message& message) {
	if (message.size() > MAX_PAYLOAD_BYTES) {
		return false;
	}

	std::lock_guard<std::mutex> lock(m_mutex);
	switch (m_state) {
	case State::Idle:
		return false;
	case State::Connecting:
		m_pending.push_back(message);
		return true;
	case State::Connected:
		m_connection->send(message);
		return true;
	}
	return false;
}

void TcpClient::Implementation::onConnected(const std::shared_ptr<asio::ip::tcp::socket>& socket) {
	std::lock_guard<std::mutex> lock(m_mutex);
	if (socket != m_socket) {
		return; // Already let go of.
	}

	Connection::Callbacks callbacks;
	callbacks.onMessage    = [this](Connection& connection, const Message& message) { onConnectionMessage(connection, message); };
	callbacks.onDisconnect = [this](Connection& connection) { onConnectionEnded(connection); };

	m_connection = std::make_shared<Connection>(std::move(*socket), CONNECTION_ID, std::move(callbacks));
	m_connection->start();
	for (const auto& message: m_pending) {
		m_connection->send(message);
	}
	m_pending.clear();
	m_resolver.reset();
	m_socket.reset();
	m_state = State::Connected;
}

void TcpClient::Implementation::onConnectFailed(const std::shared_ptr<asio::ip::tcp::socket>& socket) {
	std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		if (socket != m_socket) {
			return; // Already let go of.
		}
		m_resolver.reset();
		m_socket.reset();
		m_pending.clear();
		m_state = State::Idle;
	}

	if (m_callbacks.onDisconnect) {
		m_callbacks.onDisconnect(DisconnectReason::ConnectFailed);
	}
}

void TcpClient::Implementation::onConnectionMessage(const Connection& connection, const Message& message) {
	std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
	if (!isCurrent(connection)) {
		return;
	}

	if (m_callbacks.onMessage) {
		m_callbacks.onMessage(message);
	}
}

void TcpClient::Implementation::onConnectionEnded(const Connection& connection) {
	std::lock_guard<std::recursive_mutex> callbackLock(m_callbackMutex);
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_connection.get() != &connection) {
			return; // Already let go of.
		}
		m_connection.reset(); // Its handler still holds it.
		m_state = State::Idle;
	}

	if (m_callbacks.onDisconnect) {
		m_callbacks.onDisconnect(DisconnectReason::ConnectionLost);
	}
}

bool TcpClient::Implementation::isCurrent(const Connection& connection) const {
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_connection.get() == &connection;
}

bool TcpClient::Implementation::isIoThread() const {
	return std::this_thread::get_id() == m_ioThread.get_id();
}


TcpClient::TcpClient(Callbacks callbacks) : m_pimpl(std::make_unique<Implementation>(std::move(callbacks))) {
}

TcpClient::~TcpClient() = default;

bool TcpClient::connect(std::string host, std::uint16_t port, std::optional<Message> firstMessage) {
	return m_pimpl->connect(std::move(host), port, std::move(firstMessage));
}

bool TcpClient::disconnect() {
	return m_pimpl->disconnect();
}

bool TcpClient::isConnected() const {
	return m_pimpl->isConnected();
}

bool TcpClient::send(const Message& message) {
	return m_pimpl->send(message);
}

} // namespace network::core
} // namespace tengen
