
#pragma once

#include "network/server.hpp"
#include <gtest/gtest.h>

namespace tengen::gtest {

class MockServer : public network::IServerHandler {
public:
	MockServer();
	~MockServer();

	void onPlayerJoined(Player player) override;
	void onPlayerLeft(Player player) override;
	void onPlace(Player player, Coord c) override;
	void onPass(Player player) override;
	void onResign(Player player) override;
	void onChat(Player player, const std::string& message) override;

private:
	network::Server m_network;
	unsigned m_seated{0u}; //!< Players currently seated.
	unsigned m_turn{0u};
	unsigned m_messageId{0u};
};

} // namespace tengen::gtest
