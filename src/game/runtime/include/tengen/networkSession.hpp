#pragma once

#include "network/client.hpp"
#include "tengen/IAppSignal.hpp"
#include "tengen/IChatSession.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/position.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tengen::app {
class GameServer;

//! Gets game stat delta and constructs a local representation of the game.
//! Listeners can subscribe to certain signals, get notification when happens.
//! Listeners then check which signal and query the updated data from this NetworkSession.
//! NetworkSession is the local source of truth about the game state, GUI is just dumb renderer of this state.
class NetworkSession : public network::IClientHandler, public IGameSession, public IChatSession {
public:
	NetworkSession(const IDispatcher& dispatcher);
	~NetworkSession();

	void subscribe(IAppSignalListener* listener, uint64_t signalMask) override;
	void unsubscribe(IAppSignalListener* listener) override;

	// TODO: Maybe the UI elements should have a const reference to 'Position'. (Position is data layer; NetworkSession is application layer)
	//       Then position only has public getters and NetworkSession is a friend so it can update.
	//       Then we could remove these getters.
	//       NetworkSession updates Position. Position emits signals. Listeners query position for new data.
	// Getters
	GameStatus status() const override;
	Board board() const override;
	Player currentPlayer() const override;

	void tryPlace(unsigned x, unsigned y) override;
	void tryResign() override;
	void tryPass() override;
	void shutdown() override;

	// Network interface
	void connect(const std::string& hostIp);
	void host(unsigned boardSize);
	void disconnect();

	// Chat
	void chat(const std::string& message) override;
	std::vector<ChatEntry> getChatSince(unsigned messageId) const override;

public: // Client listener handlers
	void onGameUpdate(const network::ServerDelta& event) override;
	void onGameConfig(const network::ServerGameConfig& event) override;
	void onChatMessage(const network::ServerChat& event) override;
	void onDisconnected() override;

private:
	network::Client m_network;
	EventHub m_eventHub;
	Position m_position{};

	unsigned m_expectedMessageId{1u};                        //!< Next expected chat message id.
	std::vector<ChatEntry> m_chatHistory{};                  //!< Chat history.
	std::unordered_map<unsigned, ChatEntry> m_pendingChat{}; //!< Messages received out of order.

	std::unique_ptr<GameServer> m_localServer;
	mutable std::mutex m_stateMutex;
};

} // namespace tengen::app
