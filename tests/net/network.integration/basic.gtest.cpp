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
		ASSERT_TRUE(waitFor([&] { return client.sessionId() != network::SessionId::Empty; }));
	} // The server stops and closes the connection.
	ASSERT_TRUE(waitFor([&] { return !client.isConnected(); }));

	MockServer server;
	ASSERT_TRUE(client.connect("127.0.0.1"));
	EXPECT_TRUE(waitFor([&] { return client.sessionId() != network::SessionId::Empty; }));
	client.disconnect();
}

namespace {

//! Connects again from the first onDisconnected, and stays in it long enough for the server to welcome.
class ReconnectingHandler : public network::IClientHandler {
public:
	explicit ReconnectingHandler(network::Client& client)
	    : m_client(client) {
	}

	void onWelcome(std::optional<Player>) override {
	}
	void onGameStart(const GameConfig&) override {
	}
	void onGameDelta(const GameDelta&) override {
	}
	void onGameEnd(const GameResult&) override {
	}
	void onChatMessage(const ChatMessage&) override {
	}
	void onDisconnected(network::DisconnectReason) override {
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
	EXPECT_TRUE(waitFor([&] { return client.sessionId() != network::SessionId::Empty; }));

	client.disconnect();
	EXPECT_EQ(handler.disconnects(), 2);
	EXPECT_FALSE(client.isConnected());
}

namespace {

//! Counts the chat messages, and whether the server closed the connection after them.
class ChatCounter : public network::IClientHandler {
public:
	void onWelcome(std::optional<Player>) override {
	}
	void onGameStart(const GameConfig&) override {
	}
	void onGameDelta(const GameDelta&) override {
	}
	void onGameEnd(const GameResult&) override {
	}
	void onChatMessage(const ChatMessage&) override {
		++chats;
	}
	void onDisconnected(network::DisconnectReason) override {
		chatsBeforeDisconnect = chats.load();
		disconnected          = true;
	}

	std::atomic<unsigned> chats{0};
	std::atomic<unsigned> chatsBeforeDisconnect{0};
	std::atomic<bool> disconnected{false};
};

} // namespace

//! What the server queued before stop() still reaches the client, and only then is it disconnected.
TEST(GameNet, StopSendsQueuedMessages) {
	constexpr unsigned kMessages = 100u;

	network::Server server;
	ASSERT_TRUE(server.start());

	ChatCounter handler;
	network::Client client;
	ASSERT_TRUE(client.registerHandler(&handler));
	ASSERT_TRUE(client.connect("127.0.0.1"));
	ASSERT_TRUE(waitFor([&] { return client.isConnected(); }));

	for (unsigned i = 1u; i <= kMessages; ++i) {
		ASSERT_TRUE(server.broadcast(network::ServerChat{{.messageId = i, .player = Player::Black, .text = "last words"}}));
	}
	server.stop();
	EXPECT_FALSE(server.broadcast(network::ServerChat{{.messageId = kMessages + 1u, .player = Player::Black, .text = "too late"}}));

	ASSERT_TRUE(waitFor([&] { return handler.disconnected.load(); }));
	EXPECT_EQ(handler.chatsBeforeDisconnect, kMessages);
}

//! A second server on a taken port says so in start(), stays stopped, and leaves the first one working.
TEST(GameNet, StartFailsOnTakenPort) {
	network::Server first;
	ASSERT_TRUE(first.start());

	network::Server second;
	EXPECT_FALSE(second.start());
	EXPECT_FALSE(second.broadcast(network::ServerChat{{.messageId = 1u, .player = Player::Black, .text = "nobody hears this"}}));
	second.stop();

	network::Client client;
	ASSERT_TRUE(client.connect("127.0.0.1"));
	EXPECT_TRUE(waitFor([&] { return client.isConnected(); }));
	client.disconnect();
}

} // namespace tengen::gtest
