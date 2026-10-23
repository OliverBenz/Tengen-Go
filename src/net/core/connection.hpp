#pragma once

#include "network/core/protocol.hpp"

#include <asio.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>

namespace tengen {
namespace network::core {

//! Transportation primitive. Handles read/write of a single connection.
//! \note Internals are async and run on the IO thread of the TcpServer or TcpClient that owns it.
//!       We use shared_from_this() so any in-flight async op keeps the Connection alive.
class Connection : public std::enable_shared_from_this<Connection> {
public:
	struct Callbacks {
		std::function<void(Connection&)> onConnect;
		std::function<void(Connection&, Message)> onMessage; //!< Gets the message to keep.
		std::function<void(Connection&)> onDisconnect;
	};

	Connection(asio::ip::tcp::socket socket, ConnectionId connectionId, Callbacks callbacks);
	~Connection();

	void start();           //!< Start connection: begins async read loop and triggers onConnect.
	void stop();            //!< Stop connection: closes the socket and cancels IO (best-effort).
	void close();           //!< Close the connection once the queued messages are sent. Triggers onDisconnect. Safe to call from any thread.
	void send(Message msg); //!< Send a message to the other side. Safe to call from any thread. Move in what you don't need anymore.

	ConnectionId connectionId() const; //!< Get the identifier of this connection.

private:
	void startRead();    //!< Prime async read and dispatch messages.
	void startWrite();   //!< Prime async write for new messages.
	void doDisconnect(); //!< Internal cleanup.

private:
	std::atomic<bool> m_running{false};           //!< Connection running.
	asio::ip::tcp::socket m_socket;               //!< The connected socket.
	asio::strand<asio::any_io_executor> m_strand; //!< IO with different threads.

	ConnectionId m_connectionId; //!< Identifies the connection to its owner.
	Callbacks m_callbacks;       //!< Used to signal to the parent.

	std::deque<Message> m_writeQueue;
	bool m_writeInProgress{false};
	bool m_closing{false}; //!< close() was called: accept no more messages, disconnect once the queue is empty.
};

} // namespace network::core
} // namespace tengen
