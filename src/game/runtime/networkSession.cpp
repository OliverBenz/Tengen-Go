#include "tengen/networkSession.hpp"

#include "logging.hpp"
#include "model/gameConfig.hpp"
#include "model/gameDelta.hpp"
#include "model/gameResult.hpp"
#include "tengen/gameServer.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace tengen::app {

//! Shown until the host's config arrives with the game start.
static const GameConfig PLACEHOLDER_CONFIG{9u, fromRuleSet(RuleSet::Japanese)};

NetworkSession::NetworkSession() {
	m_network.registerHandler(this);
}
NetworkSession::~NetworkSession() {
	disconnect();
}

void NetworkSession::subscribe(IAppSignalListener* listener, uint64_t signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void NetworkSession::unsubscribe(IAppSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}


void NetworkSession::connect(const std::string& hostIp) {
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_gameInfo.reset(PLACEHOLDER_CONFIG, GameStatus::Ready);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}
	m_localServer.reset();
	m_network.connect(hostIp);

	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}

bool NetworkSession::host(const GameConfig& config, const Player hostColour) {
	disconnect();

	// Creating server may throw on invalid game config.
	try {
		m_localServer = std::make_unique<GameServer>(config, hostColour);
	} catch (const std::invalid_argument& e) {
		Logger().Log(Logging::LogLevel::Error, std::format("[NetworkSession] Cannot host the game: {}", e.what()));
		return false;
	}

	// Initialize the session
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_gameInfo.reset(config, GameStatus::Ready);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}

	// We connect first, so we take the seat the server hands to its first player.
	m_localServer->start();
	m_network.connect("127.0.0.1");

	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);

	return true;
}

void NetworkSession::disconnect() {
	m_network.disconnect();
	if (m_localServer) {
		m_localServer->stop();
		m_localServer.reset();
	}

	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_gameInfo.reset(PLACEHOLDER_CONFIG, GameStatus::Idle);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}

	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}

void NetworkSession::shutdown() {
	if (m_localServer) {
		m_localServer->stop();
		m_localServer.reset();
	}
	m_network.disconnect();

	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_gameInfo.reset(PLACEHOLDER_CONFIG, GameStatus::Idle);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}
}


void NetworkSession::tryPlace(unsigned x, unsigned y) {
	m_network.send(network::ClientPutStone{.c = {x, y}});
}
void NetworkSession::tryResign() {
	m_network.send(network::ClientResign{});
}
void NetworkSession::tryPass() {
	m_network.send(network::ClientPass{});
}
void NetworkSession::chat(const std::string& message) {
	m_network.send(network::ClientChat{message});
}

GameStatus NetworkSession::status() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getStatus();
}
Board NetworkSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getBoard();
}
Player NetworkSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getPlayer();
}
std::vector<ChatEntry> NetworkSession::getChatSince(const unsigned messageId) const {
	std::lock_guard<std::mutex> lock(m_stateMutex);

	// Find first entry with id > messageId
	auto it = std::upper_bound(m_chatHistory.begin(), m_chatHistory.end(), messageId,
	                           [](const unsigned value, const ChatEntry& e) { return e.messageId > value; });

	return {it, m_chatHistory.end()};
}

void NetworkSession::onGameStart(const GameConfig& config) {
	bool started = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		started = m_gameInfo.update(config);
	}
	if (!started) {
		return;
	}
	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}
void NetworkSession::onGameDelta(const GameDelta& delta) {
	bool applied = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		applied = m_gameInfo.update(delta);
	}

	if (!applied) {
		return;
	}

	switch (delta.action) {
	case GameAction::Place:
		m_eventHub.signal(AS_BoardChange);
		m_eventHub.signal(AS_PlayerChange);
		break;
	case GameAction::Pass:
		m_eventHub.signal(AS_PlayerChange);
		break;
	case GameAction::Resign:
		break;
	}
}
void NetworkSession::onGameEnd(const GameResult& result) {
	bool ended = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		ended = m_gameInfo.update(result);
	}
	if (!ended) {
		return;
	}
	m_eventHub.signal(AS_StateChange);
}
void NetworkSession::onChatMessage(const Player player, const unsigned messageId, const std::string& message) {
	bool appended = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);

		if (messageId < m_expectedMessageId) {
			// Ignore already seen messages.
		} else if (messageId == m_expectedMessageId) {
			m_chatHistory.emplace_back(ChatEntry{player, messageId, message});
			++m_expectedMessageId;
			appended = true;
		} else {
			m_pendingChat.emplace(messageId, ChatEntry{player, messageId, message});
		}

		// Try insterting pending chat messages to history.
		while (true) {
			auto it = m_pendingChat.find(m_expectedMessageId);
			if (it == m_pendingChat.end()) {
				break;
			}
			m_chatHistory.emplace_back(it->second);
			m_pendingChat.erase(it);
			++m_expectedMessageId;
			appended = true;
		}
	}
	if (appended) {
		m_eventHub.signal(AS_NewChat);
	}
}
void NetworkSession::onDisconnected() {
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_gameInfo.reset(PLACEHOLDER_CONFIG, GameStatus::Idle);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}
	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}

} // namespace tengen::app
