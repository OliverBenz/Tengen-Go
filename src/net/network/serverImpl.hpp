#pragma once

#include "network/server.hpp"

#include "SafeQueue.hpp"
#include "network/core/tcpServer.hpp"
#include "serverEvents.hpp"
#include "sessionManager.hpp"

#include <cstdint>
#include <mutex>
#include <thread>

namespace tengen::network {

class Server::Implementation {
public:
	explicit Implementation(std::uint16_t port);

	bool start();
	void stop();
	bool registerHandler(IServerHandler* handler);
	void setFirstSeat(Seat seat);

	bool send(SessionId sessionId, const ServerEvent& event); //!< Queue an event for one client. Any thread.
	bool broadcast(const ServerEvent& event);                 //!< Queue an event for all welcomed clients. Any thread.

private:
	void serverLoop(); //!< Server thread: drain queue and act.

	Seat freeSeat() const;

	bool queue(ServerQueueEvent event); //!< Queue any event while running. Dropped once stopped: the server thread won't handle it.

	// Server thread: the only one that reads the sessions.
	void sendNow(SessionId sessionId, const core::Message& message);
	void broadcastNow(const core::Message& message);

private:
	// Network callbacks (run on libNetwork threads) just enqueue events.
	void onClientConnected(core::ConnectionId connectionId);
	void onClientMessage(core::ConnectionId connectionId, core::Message payload);
	void onClientDisconnected(core::ConnectionId connectionId);

private:
	// Processing of server events. The server loop hands each event to its overload.
	void processEvent(const ClientConnected& event);                  //!< Creates session key.
	void processEvent(const ClientDisconnected& event);               //!< Destroys session key.
	void processEvent(const ClientMessage& event);                    //!< Translate payload to network event and handle.
	void processEvent(const Send& event);                             //!< Send to one client.
	void processEvent(const Broadcast& event);                        //!< Send to all welcomed clients.
	void processEvent(const Shutdown& event);                         //!< Never reached: serverLoop ends on it.
	void processHello(SessionId sessionId, const ClientHello& hello); //!< Welcome the client with a seat, or reject it.

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

	IServerHandler* m_handler{nullptr};       //!< The class that will handle server events.
	Seat m_firstSeat{Seat::Black};            //!< Seat handed out first while both are free.
	SafeQueue<ServerQueueEvent> m_eventQueue; //!< Event queue between network threads and server thread.

	core::TcpServer m_network; //!< Keep last: its callbacks use the members above, so it must be created after them and destroyed before them.
};

} // namespace tengen::network
