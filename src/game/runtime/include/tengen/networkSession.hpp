#pragma once

#include "model/gameConfig.hpp"
#include "network/client.hpp"
#include "tengen/IAppSignal.hpp"
#include "tengen/IChatSession.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/sessionGameInfo.hpp"

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
	NetworkSession();
	~NetworkSession();

	void connect(const std::string& hostIp);
	bool host(const GameConfig& config, Player hostColour); //!< Host a game and join it as hostColour. Returns false on invalid game configs.
	void disconnect();

public: // IAppSignalSource Interface
	void subscribe(IAppSignalListener* listener, uint64_t signalMask) override;
	void unsubscribe(IAppSignalListener* listener) override;

public: // IGameSession Interface
	GameStatus status() const override;
	Board board() const override;
	Player currentPlayer() const override;

	void tryPlace(unsigned x, unsigned y) override;
	void tryResign() override;
	void tryPass() override;
	void shutdown() override;

public: // IChatSession Interface
	void chat(const std::string& message) override;
	std::vector<ChatEntry> getChatSince(unsigned messageId) const override;

public: // network::IClientHandler Interface
	void onGameStart(const GameConfig& config) override;
	void onGameDelta(const GameDelta& delta) override;
	void onGameEnd(const GameResult& result) override;
	void onChatMessage(Player player, unsigned messageId, const std::string& message) override;
	void onDisconnected() override;

private:
	network::Client m_network;
	EventHub m_eventHub;
	SessionGameInfo m_gameInfo{};

	unsigned m_expectedMessageId{1u};                        //!< Next expected chat message id.
	std::vector<ChatEntry> m_chatHistory{};                  //!< Chat history.
	std::unordered_map<unsigned, ChatEntry> m_pendingChat{}; //!< Messages received out of order.

	std::unique_ptr<GameServer> m_localServer;
	mutable std::mutex m_stateMutex;
};

} // namespace tengen::app
