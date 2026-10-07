#include "network/server.hpp"
#include "network/client.hpp"
#include "network/nwEvents.hpp"
#include "network/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace tengen::gtest {

class TestClientHandler final : public network::IClientHandler {
public:
	void onGameStart(const GameConfig&) override {
	}

	void onGameDelta(const GameDelta& delta) override {
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_lastDelta = delta;
		}
		m_cv.notify_all();
	}

	void onGameEnd(const GameResult&) override {
	}

	void onChatMessage(Player, unsigned, const std::string&) override {
	}

	void onDisconnected() override {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_disconnected = true;
		m_cv.notify_all();
	}

	bool waitForDelta(std::chrono::milliseconds timeout, GameDelta& out) {
		std::unique_lock<std::mutex> lock(m_mutex);
		if (!m_cv.wait_for(lock, timeout, [&] { return m_lastDelta.has_value() || m_disconnected; })) {
			return false;
		}
		if (!m_lastDelta.has_value()) {
			return false;
		}
		out = *m_lastDelta;
		return true;
	}

private:
	std::mutex m_mutex;
	std::condition_variable m_cv;
	std::optional<GameDelta> m_lastDelta;
	bool m_disconnected{false};
};

class TestServerHandler final : public network::IServerHandler {
public:
	explicit TestServerHandler(network::Server& server) : m_server(server) {
	}

	void onPlayerJoined(Player) override {
	}
	void onPlayerLeft(Player) override {
	}

	void onPlace(const Player player, const Coord c) override {
		m_server.broadcast(network::ServerGameDelta{GameDelta{
		        .moveId     = ++m_turn,
		        .action     = GameAction::Place,
		        .player     = player,
		        .coord      = c,
		        .captures   = {},
		        .nextPlayer = opponent(player),
		}});
	}

	void onPass(Player) override {
	}
	void onResign(Player) override {
	}
	void onChat(Player, const std::string&) override {
	}

private:
	network::Server& m_server;
	std::atomic<unsigned> m_turn{0};
};

TEST(Networking, ServerDeltaFromPutStone) {
	constexpr std::uint16_t kPort = 12346;

	network::Server server{kPort};
	TestServerHandler serverHandler(server);
	ASSERT_TRUE(server.registerHandler(&serverHandler));
	server.start();

	network::Client client1;
	network::Client client2;
	TestClientHandler handler1;
	TestClientHandler handler2;

	ASSERT_TRUE(client1.registerHandler(&handler1));
	ASSERT_TRUE(client2.registerHandler(&handler2));

	client1.connect("127.0.0.1", kPort);
	client2.connect("127.0.0.1", kPort);

	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// Player 1 places a stone. Valid move
	ASSERT_TRUE(client1.send(network::ClientPutStone{1u, 2u}));

	// Player 2 recives the delta
	GameDelta delta{};
	ASSERT_TRUE(handler2.waitForDelta(std::chrono::milliseconds(300), delta));

	EXPECT_EQ(delta.moveId, 1u);
	EXPECT_EQ(delta.player, Player::Black);
	EXPECT_EQ(delta.action, GameAction::Place);
	ASSERT_TRUE(delta.coord.has_value());
	EXPECT_EQ(delta.coord->x, 1u);
	EXPECT_EQ(delta.coord->y, 2u);
	EXPECT_EQ(delta.captures.size(), 0u);
	EXPECT_EQ(delta.nextPlayer, Player::White);

	client1.disconnect();
	client2.disconnect();
	server.stop();
}

} // namespace tengen::gtest
