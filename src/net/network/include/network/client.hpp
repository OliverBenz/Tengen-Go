#pragma once

#include "network/nwEvents.hpp"
#include "network/types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace tengen::network {

//! Callback interface invoked on the client's read thread. Gets the game as model data.
//! \note Keep handlers lightweight.
class IClientHandler {
public:
	virtual ~IClientHandler()                                                                 = default;
	virtual void onGameStart(const GameConfig& config)                                        = 0; //!< The game started.
	virtual void onGameDelta(const GameDelta& delta)                                          = 0; //!< One accepted move.
	virtual void onGameEnd(const GameResult& result)                                          = 0; //!< The game ended. Right after the last delta.
	virtual void onChatMessage(Player player, unsigned messageId, const std::string& message) = 0; //!< A chat message is received.
	virtual void onDisconnected()                                                             = 0; //!< You disconnected from the server.
};

class Client {
public:
	Client();
	~Client();

	Client(const Client&)            = delete;
	Client& operator=(const Client&) = delete;
	Client(Client&&)                 = delete;
	Client& operator=(Client&&)      = delete;

	bool registerHandler(IClientHandler* handler); //!< Register a single handler. Returns false if already registered.

	bool connect(const std::string& host);                     //!< Connect to server using default port.
	bool connect(const std::string& host, std::uint16_t port); //!< Connect to server using a custom port.
	void disconnect();                                         //!< Disconnect from the server.
	bool isConnected() const;                                  //!< Check if connected to a server.


	bool send(const ClientEvent& event); //!< Send a client event to the server. Returns false on failure.
	SessionId sessionId() const;         //!< Session id assigned by server. 0 means unassigned.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide networking protocol stuff.
};

} // namespace tengen::network
