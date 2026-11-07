#include "mockServer.hpp"

#include <format>

namespace tengen::gtest {

MockServer::MockServer() {
	EXPECT_TRUE(m_network.registerHandler(this));
	EXPECT_TRUE(m_network.start());
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
	        .move     = {.moveId = ++m_turn, .player = player, .action = Place{c}},
	        .captures = {},
	}});
}

void MockServer::onPass(const Player player) {
	m_network.broadcast(network::ServerGameDelta{GameDelta{
	        .move     = {.moveId = ++m_turn, .player = player, .action = Pass{}},
	        .captures = {},
	}});
}

void MockServer::onResign(const Player player) {
	m_network.broadcast(network::ServerGameEnd{GameResult{.winner = opponent(player), .reason = EndReason::Resignation}});
}

void MockServer::onChat(const Player player, const std::string& message) {
	m_network.broadcast(network::ServerChat{{.messageId = m_messageId++, .player = player, .text = message}});
}

} // namespace tengen::gtest
