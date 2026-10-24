#pragma once

#include "network/nwEvents.hpp"
#include "network/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tengen::network {

//! Why a connection ended.
enum class DisconnectReason : std::uint8_t {
	ClosedLocally,    //!< disconnect() was called.
	ConnectFailed,    //!< The server could not be reached.
	ConnectionLost,   //!< The server closed the connection, or it broke.
	ProtocolMismatch, //!< Rejected: the server speaks another version.
	ProtocolError,    //!< The server sent something it shouldn't.
	Error,            //!< Handling a message threw.
};

//! Gets the game from the server as model data.
//! Calls come one at a time on the client's IO thread. Only onDisconnected after your own disconnect() comes on your thread.
//! Keep handlers short, and never wait in one for the thread that connects, disconnects or destroys the client: both would hang.
class IClientHandler {
public:
	virtual ~IClientHandler()                                                                 = default;
	virtual void onWelcome(std::optional<Player> player)                                      = 0; //!< The server took us in. Nullopt: we only watch.
	virtual void onGameStart(const GameConfig& config)                                        = 0; //!< The game started.
	virtual void onGameDelta(const GameDelta& delta)                                          = 0; //!< One accepted move.
	virtual void onGameEnd(const GameResult& result)                                          = 0; //!< The game ended. Right after the last delta.
	virtual void onChatMessage(Player player, unsigned messageId, const std::string& message) = 0; //!< A chat message is received.
	virtual void onDisconnected(DisconnectReason reason)                                      = 0; //!< The connection ended, or could not be made.
};

class Client {
public:
	Client();
	~Client(); //!< Ends the connection without calling the handler.

	Client(const Client&)            = delete;
	Client& operator=(const Client&) = delete;
	Client(Client&&)                 = delete;
	Client& operator=(Client&&)      = delete;

	bool registerHandler(IClientHandler* handler); //!< Once, before connect(). False if one is set already.

	bool connect(const std::string& host);                     //!< Start connecting. False if already connected or connecting.
	bool connect(const std::string& host, std::uint16_t port); //!< Start connecting. False if already connected or connecting.
	void disconnect();                                         //!< If there was a connection, calls onDisconnected before it returns.
	bool isConnected() const;                                  //!< True once the server welcomed us.

	bool send(const ClientEvent& event); //!< Sends, or queues until connected. False if not connected or connecting.
	SessionId sessionId() const;         //!< Session id assigned by server. 0 until the welcome, and again after onDisconnected.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide networking protocol stuff.
};

} // namespace tengen::network
