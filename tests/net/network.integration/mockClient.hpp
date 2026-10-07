#pragma once

#include "network/client.hpp"

namespace tengen::gtest {

class MockClient : public network::IClientHandler {
public:
	MockClient();
	~MockClient();

	void disconnect();
	void chat(const std::string& message);
	void tryPlace(unsigned x, unsigned y);

public:
	void onGameStart(const GameConfig& config) override;
	void onGameDelta(const GameDelta& delta) override;
	void onGameEnd(const GameResult& result) override;
	void onChatMessage(Player player, unsigned messageId, const std::string& message) override;
	void onDisconnected() override;

private:
	network::Client m_network;
};

} // namespace tengen::gtest
