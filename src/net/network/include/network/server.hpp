#pragma once

#include "network/nwEvents.hpp"
#include "network/types.hpp"

#include <cstdint>
#include <memory>

namespace tengen::network {

//! Callback interface invoked on the server's processing thread.
//! The Server resolves sessions and seats itself: only seated players reach the handler, named by their colour.
//! \note Keep handlers lightweight.
class IServerHandler {
public:
	virtual ~IServerHandler()                                      = default;
	virtual void onPlayerJoined(Player player)                     = 0; //!< A client took the seat of this player.
	virtual void onPlayerLeft(Player player)                       = 0; //!< The client of this player disconnected.
	virtual void onPlace(Player player, Coord c)                   = 0; //!< The player wants to place a stone at c.
	virtual void onPass(Player player)                             = 0; //!< The player wants to pass.
	virtual void onResign(Player player)                           = 0; //!< The player wants to resign.
	virtual void onChat(Player player, const std::string& message) = 0; //!< The player sent a chat message.
};

class Server {
public:
	Server();
	explicit Server(std::uint16_t port);
	~Server();

	Server(const Server&)            = delete;
	Server& operator=(const Server&) = delete;
	Server(Server&&)                 = delete;
	Server& operator=(Server&&)      = delete;

	void start();
	void stop();

	bool registerHandler(IServerHandler* handler); //!< Register a single handler. Returns false if already registered.
	void setFirstSeat(Seat seat);                  //!< Seat the first player to connect takes; the second takes the other. Black unless set. Call before start().

	bool send(SessionId sessionId, const ServerEvent& event); //!< Send event to client with given sessionId. Returns false on failure.
	bool broadcast(const ServerEvent& event);                 //!< Send event to all connected clients. Returns true if any send succeeded.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide networking protocol stuff.
};

} // namespace tengen::network
