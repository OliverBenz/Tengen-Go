#pragma once

#include "engine/engineCatalog.hpp"
#include "model/gameUpdates.hpp"
#include "model/player.hpp"
#include "tengen/IChatSession.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/INetworkSession.hpp"

#include <memory>
#include <string>
#include <variant>

namespace tengen::app {

class OpenSession;
class BotSession;
class NetworkSession;

//! Owns the one active session. Starting a session ends the one before.
//! \note GUI thread only. Release everything that holds on to the session before starting another one or shutting down.
class SessionManager {
public:
	SessionManager(); // Both defined where the sessions are complete.
	~SessionManager();

	// Start a session
	void startOpen(const GameConfig& config);
	void startBot(const GameConfig& config, const engine::EngineConfig& engineConfig, bool humanPlaysBlack); //!< \throws std::invalid_argument if the board size is not supported.
	void startNetwork();                                                                                     //!< A network session starts idle, so listeners can subscribe before anything arrives. Then connect() or host().
	void shutdown();                                                                                         //!< End the active session, if any.

	void connect(const std::string& hostIp);                //!< Join a hosted game. Needs startNetwork() first.
	bool host(const GameConfig& config, Player hostColour); //!< Host a game and join it as hostColour. False if it cannot be hosted. Needs startNetwork() first.


	IGameSession* game();       //!< Null without a session.
	IChatSession* chat();       //!< Null unless the session has a chat.
	INetworkSession* network(); //!< Null unless the session runs over the network.

private:
	using Session = std::variant<OpenSession, BotSession, NetworkSession>;
	std::unique_ptr<Session> m_session{nullptr}; //!< The active session, null without one. Behind a pointer, so the sessions stay out of this header.
};

} // namespace tengen::app
