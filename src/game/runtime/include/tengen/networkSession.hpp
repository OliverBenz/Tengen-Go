#pragma once

#include "model/gameConfig.hpp"
#include "network/client.hpp"
#include "tengen/IAppSignal.hpp"
#include "tengen/IChatSession.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/INetworkSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/sessionChatInfo.hpp"
#include "tengen/sessionGameInfo.hpp"
#include "tengen/sessionNetworkInfo.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace tengen::app {

class GameServer;

//! Gets game state delta and constructs a local representation of the game.
//! Listeners can subscribe to certain signals, get notification when happens.
//! Listeners then check which signal and query the updated data from this NetworkSession.
//! NetworkSession is the local source of truth about the game state, GUI is just dumb renderer of this state.
class NetworkSession final : public IGameSession, public IChatSession, public INetworkSession, private network::IClientHandler {
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
	std::vector<ChatMessage> getChatSince(unsigned messageId) const override;

public: // INetworkSession Interface
	SessionNetworkInfo networkInfo() const override;

private: // network::IClientHandler Interface
	void onWelcome(std::optional<Player> player) override;
	void onGameStart(const GameConfig& config) override;
	void onGameDelta(const GameDelta& delta) override;
	void onGameEnd(const GameResult& result) override;
	void onChatMessage(const ChatMessage& chat) override;
	void onDisconnected(network::DisconnectReason reason) override;

private:
	void stopLocalServer();                                       //!< Stop the hosted server off the GUI thread: it waits for slow clients.
	void resetState(const GameConfig& config, GameStatus status); //!< Forget the last game and chat. The network info has its own rules.

private:
	EventHub m_eventHub;
	SessionGameInfo m_gameInfo{};
	SessionChatInfo m_chatInfo{};
	SessionNetworkInfo m_networkInfo{};

	std::unique_ptr<GameServer> m_localServer;
	std::thread m_serverStopper;     //!< Stops the last hosted server.
	mutable std::mutex m_stateMutex; //!< Guards the three infos.

	network::Client m_network; //!< Last, so it goes first and no callback reaches the members above.
};

} // namespace tengen::app
