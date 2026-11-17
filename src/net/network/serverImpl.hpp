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
	bool queue(ServerQueueEvent event); //!< Queue an event while running. Any thread. Dropped once stopped: the server thread won't handle it.

	// Server thread: the only one that touches the sessions.
	void serverLoop();                                                //!< Handle queued events until the shutdown.
	void processEvent(const ClientConnected& event);                  //!< Creates session key.
	void processEvent(const ClientDisconnected& event);               //!< Destroys session key.
	void processEvent(const ClientMessage& event);                    //!< Translate payload to network event and handle.
	void processEvent(const Send& event);                             //!< Send to one client.
	void processEvent(const Broadcast& event);                        //!< Send to all welcomed clients.
	void processEvent(const Shutdown& event);                         //!< Never reached: serverLoop ends on it.
	void processHello(SessionId sessionId, const ClientHello& hello); //!< Welcome the client with a seat, or reject it.

	void sendNow(SessionId sessionId, const core::Message& message);
	void broadcastNow(const core::Message& message);
	Seat freeSeat() const;

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
