#include "mockClient.hpp"
#include "mockServer.hpp"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

namespace tengen::gtest {

TEST(GameNet, Connection) {
	auto server  = MockServer();
	auto client1 = MockClient();
	auto client2 = MockClient();

	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	client1.chat("Helloooooou");
	client2.tryPlace(0u, 0u);
	client1.tryPlace(1u, 2u);

	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	client1.disconnect();
	client2.disconnect();
}

namespace {

//! Poll the condition until it holds or a second passed.
template <typename Condition>
bool waitFor(Condition condition) {
	for (int i = 0; i < 100; ++i) {
		if (condition()) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	return condition();
}

} // namespace

//! A client the server dropped can connect again, and gets welcomed again.
TEST(GameNet, ConnectAgainAfterServerClosed) {
	network::Client client;
	{
		MockServer server;
		ASSERT_TRUE(client.connect("127.0.0.1"));
		ASSERT_TRUE(waitFor([&] { return client.sessionId() != 0u; }));
	} // The server stops and closes the connection.
	ASSERT_TRUE(waitFor([&] { return !client.isConnected(); }));

	MockServer server;
	ASSERT_TRUE(client.connect("127.0.0.1"));
	EXPECT_TRUE(waitFor([&] { return client.sessionId() != 0u; }));
	client.disconnect();
}

namespace {

//! Connects again from the first onDisconnected, and stays in it long enough for the server to welcome.
class ReconnectingHandler : public network::IClientHandler {
public:
	explicit ReconnectingHandler(network::Client& client) : m_client(client) {
	}

	void onGameStart(const GameConfig&) override {
	}
	void onGameDelta(const GameDelta&) override {
	}
	void onGameEnd(const GameResult&) override {
	}
	void onChatMessage(Player, unsigned, const std::string&) override {
	}
	void onDisconnected() override {
		if (++m_disconnects == 1) {
			EXPECT_TRUE(m_client.connect("127.0.0.1"));
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	int disconnects() const {
		return m_disconnects;
	}

private:
	network::Client& m_client;
	std::atomic<int> m_disconnects{0};
};

} // namespace

//! A session opened from onDisconnected survives the end of the old one being reported.
TEST(GameNet, ConnectAgainFromOnDisconnected) {
	MockServer server;
	network::Client client;
	ReconnectingHandler handler(client);
	ASSERT_TRUE(client.registerHandler(&handler));
	ASSERT_TRUE(client.connect("127.0.0.1"));
	ASSERT_TRUE(waitFor([&] { return client.isConnected(); }));

	client.disconnect();
	EXPECT_EQ(handler.disconnects(), 1);
	EXPECT_TRUE(waitFor([&] { return client.sessionId() != 0u; }));

	client.disconnect();
	EXPECT_EQ(handler.disconnects(), 2);
	EXPECT_FALSE(client.isConnected());
}

} // namespace tengen::gtest
