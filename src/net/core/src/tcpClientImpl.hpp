#pragma once

#include "network/core/tcpClient.hpp"

#include "connection.hpp"

#include <asio.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace tengen {
namespace network::core {

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

} // namespace network::core
} // namespace tengen
