#include "mockClient.hpp"

#include <format>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>

namespace tengen::gtest {

MockClient::MockClient() {
	EXPECT_TRUE(m_network.registerHandler(this));
	m_network.connect("127.0.0.1");
}

MockClient::~MockClient() {
	disconnect();
}

void MockClient::disconnect() {
	m_network.disconnect();
}

void MockClient::chat(const std::string& message) {
	m_network.send(network::ClientChat{message});
}

void MockClient::tryPlace(unsigned x, unsigned y) {
	m_network.send(network::ClientPutStone{.c = {x, y}});
}

void MockClient::onGameStart(const GameConfig& config) {
	std::cout << std::format("[Client] Received game start: board={}, komi={}\n", config.boardSize, config.rules.komi);
}

void MockClient::onGameDelta(const GameDelta& delta) {
	const auto player = toString(delta.player);
	switch (delta.action) {
	case GameAction::Place:
		if (delta.coord.has_value()) {
			std::cout << std::format("[Client] Received board update from '{}' at ({}, {}).\n", player, delta.coord->x, delta.coord->y);
		} else {
			std::cout << std::format("[Client] Received board update from '{}'.\n", player);
		}
		break;
	case GameAction::Pass:
		std::cout << std::format("[Client] Received pass from '{}'.\n", player);
		break;
	case GameAction::Resign:
		std::cout << std::format("[Client] Received resign from '{}'\n", player);
		break;
	}
}

void MockClient::onGameEnd(const GameResult& result) {
	std::cout << std::format("[Client] Received game end. Winner: {}\n", result.winner ? toString(*result.winner) : "none");
}

void MockClient::onChatMessage(const Player player, unsigned, const std::string& message) {
	std::cout << std::format("[Client] Received message from '{}':{}\n", toString(player), message);
}

void MockClient::onDisconnected() {
	std::cout << std::format("[Client] Client {} disconnected.\n", m_network.sessionId());
}

} // namespace tengen::gtest
