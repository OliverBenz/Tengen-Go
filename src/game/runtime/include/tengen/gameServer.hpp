#pragma once

#include "core/IGameStateListener.hpp"
#include "core/game.hpp"
#include "model/gameConfig.hpp"
#include "model/player.hpp"
#include "network/server.hpp"

#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace tengen {
namespace app {


//! Hosts a game for two network players and handles the communication between network and game.
class GameServer : public network::IServerHandler, public IGameStateListener {
public:
	//! The first client to connect plays firstPlayer.
	//! \throws std::invalid_argument if the board size is not supported.
	GameServer(const GameConfig& config, Player firstPlayer);
	~GameServer();

	void start(); //!< Boot the game loop, the network listener and the server event loop.
	void stop();  //!< Signal shutdown to the server loop and stop the network listener.

public: // IServerHandler Interface
	void onPlayerJoined(Player player) override;
	void onPlayerLeft(Player player) override;
	void onPlace(Player player, Coord c) override;
	void onPass(Player player) override;
	void onResign(Player player) override;
	void onChat(Player player, const std::string& message) override;

public: // IGameStateListener Interface
	void onGameStart(const GameConfig& config) override;
	void onGameDelta(const GameDelta& delta) override;
	void onGameEnd(const GameResult& result) override;

private:
	struct ChatEntry {
		Player player;
		std::string message;
	};

private:
	Game m_game;              //!< Our unique game instance.
	std::thread m_gameThread; //!< Runs the game loop.

	std::unordered_set<Player> m_seated;  //!< Players whose seat is taken.
	std::vector<ChatEntry> m_chatHistory; //!< Well what could that be.

	network::Server m_server{}; //!< The network interface.
};

} // namespace app
} // namespace tengen
