#include "mockServer.hpp"

#include <format>

namespace tengen::gtest {

MockServer::MockServer() {
	EXPECT_TRUE(m_network.registerHandler(this));
	m_network.start();
}
MockServer::~MockServer() {
	m_network.stop();
}

void MockServer::onPlayerJoined(const Player player) {
	std::cout << std::format("[Server] {} joined.\n", toString(player));

	// Like the real server, the game starts once both seats are taken.
	if (++m_seated == 2u) {
		m_network.broadcast(network::ServerGameStart{GameConfig{.boardSize = 9u, .rules = fromRuleSet(RuleSet::Japanese)}});
	}
}

void MockServer::onPlayerLeft(const Player player) {
	std::cout << std::format("[Server] {} left.\n", toString(player));
	--m_seated;
}

void MockServer::onPlace(const Player player, const Coord c) {
	m_network.broadcast(network::ServerGameDelta{GameDelta{
	        .moveId     = ++m_turn,
	        .action     = GameAction::Place,
	        .player     = player,
	        .coord      = c,
	        .captures   = {},
	        .nextPlayer = opponent(player),
	}});
}

void MockServer::onPass(const Player player) {
	m_network.broadcast(network::ServerGameDelta{GameDelta{
	        .moveId     = ++m_turn,
	        .action     = GameAction::Pass,
	        .player     = player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = opponent(player),
	}});
}

void MockServer::onResign(const Player player) {
	m_network.broadcast(network::ServerGameDelta{GameDelta{
	        .moveId     = ++m_turn,
	        .action     = GameAction::Resign,
	        .player     = player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = opponent(player),
	}});
	m_network.broadcast(network::ServerGameEnd{GameResult{.winner = opponent(player), .reason = EndReason::Resignation}});
}

void MockServer::onChat(const Player player, const std::string& message) {
	m_network.broadcast(network::ServerChat{player, m_messageId++, message});
}

} // namespace tengen::gtest
