#pragma once

#include "network/core/tcpServer.hpp"

#include "connection.hpp"

#include <asio.hpp>
#include <asio/ip/tcp.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace tengen {
namespace network::core {

class TcpServer::Implementation {
public:
	Implementation(Callbacks callbacks, std::uint16_t port);

	bool start();
	void stop();

	bool send(ConnectionId connectionId, const Message& msg);
	void disconnect(ConnectionId connectionId);

private:
	enum class State : std::uint8_t {
		Created,
		Running,
		Stopped, //!< For good: a stopped server can't start again.
	};

	void setupAcceptor(std::uint16_t port);
	void doAccept();                 //!< Start async accept loop.
	void retryAccept();              //!< Start the next accept after a pause.
	ConnectionId nextConnectionId(); //!< Never Empty. Only used on the IO thread.

	std::shared_ptr<Connection> createConnection(asio::ip::tcp::socket socket, ConnectionId connectionId); //!< Create and add a new connection to the map. Null if that fails. Needs m_connectionsMutex.

private:
	asio::io_context m_ioContext{};                     //!< Main context. Initialized: The acceptor uses it in the ctor initializer.
	asio::ip::tcp::acceptor m_acceptor;                 //!< Accept tcp connections on the IO thread.
	asio::steady_timer m_acceptRetryTimer{m_ioContext}; //!< Only used on the IO thread.
	std::optional<asio::executor_work_guard<asio::io_context::executor_type>> m_workGuard;
	bool m_acceptorReady{false};

	std::thread m_ioThread;                     //!< IO context thread.
	std::atomic<State> m_state{State::Created}; //!< Read on the IO thread.
	std::mutex m_lifecycleMutex;                //!< start() and stop() run one at a time, so stop() always finds the thread start() made. Also makes a second stop() wait for the first.

	const Callbacks m_callbacks; //!< Callback functions to signal events. Set once, so the IO thread can read them without a lock.

	std::underlying_type_t<ConnectionId> m_nextConnectionId{1u};                 //!< Only used on the IO thread.
	std::unordered_map<ConnectionId, std::shared_ptr<Connection>> m_connections; //!< Active connections.
	std::mutex m_connectionsMutex;                                               //!< Handle concurrency.
	std::condition_variable m_connectionsGone;                                   //!< The last connection left the map. stop() waits for it.
};

} // namespace network::core
} // namespace tengen
