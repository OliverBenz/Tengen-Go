#include "tengen/networkSession.hpp"

#include "logging.hpp"
#include "tengen/gameServer.hpp"

#include <algorithm>
#include <cassert>

namespace tengen::app {

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
		m_position.reset(9u);
		m_position.setStatus(GameStatus::Ready);
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

void NetworkSession::host(unsigned boardSize) {
	disconnect();

	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_position.reset(boardSize);
		m_position.setStatus(GameStatus::Ready);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}

	m_localServer = std::make_unique<GameServer>(boardSize);
	m_localServer->start();
	m_network.connect("127.0.0.1");

	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}

void NetworkSession::disconnect() {
	m_network.disconnect();
	if (m_localServer) {
		m_localServer->stop();
		m_localServer.reset();
	}

	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_position.reset(9u);
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
		m_position.reset(9u);
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
	return m_position.getStatus();
}
Board NetworkSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getBoard();
}
Player NetworkSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getPlayer();
}
std::vector<ChatEntry> NetworkSession::getChatSince(const unsigned messageId) const {
	std::lock_guard<std::mutex> lock(m_stateMutex);

	// Find first entry with id > messageId
	auto it = std::upper_bound(m_chatHistory.begin(), m_chatHistory.end(), messageId,
	                           [](const unsigned value, const ChatEntry& e) { return e.messageId > value; });

	return {it, m_chatHistory.end()};
}

void NetworkSession::onGameUpdate(const network::ServerDelta& event) {
	GameStatus status         = GameStatus::Active;
	GameStatus previousStatus = GameStatus::Active;
	bool applied              = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		previousStatus = m_position.getStatus();
		applied        = m_position.apply(event);
		status         = m_position.getStatus(); // For signalling later
	}

	if (!applied) {
		return;
	}

	// Signalling depending on action
	switch (event.action) {
	case network::ServerAction::Place:
		m_eventHub.signal(AS_BoardChange);
		m_eventHub.signal(AS_PlayerChange);
		break;
	case network::ServerAction::Pass:
		m_eventHub.signal(AS_PlayerChange);
		break;
	case network::ServerAction::Resign:
		break;
	case network::ServerAction::Count:
		assert(false); //!< This should already be prohibited by libGameNet.
		break;
	};
	if (previousStatus != status) {
		m_eventHub.signal(AS_StateChange);
	}
}
void NetworkSession::onGameConfig(const network::ServerGameConfig& event) {
	bool initialized = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		initialized = m_position.init(event);
	}
	if (!initialized) {
		return;
	}
	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}
void NetworkSession::onChatMessage(const network::ServerChat& event) {
	bool appended = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);

		if (event.messageId < m_expectedMessageId) {
			// Ignore already seen messages.
		} else if (event.messageId == m_expectedMessageId) {
			m_chatHistory.emplace_back(ChatEntry{event.player, event.messageId, event.message});
			++m_expectedMessageId;
			appended = true;
		} else {
			m_pendingChat.emplace(event.messageId, ChatEntry{event.player, event.messageId, event.message});
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
		m_position.reset(9u);
		m_expectedMessageId = 1u;
		m_chatHistory.clear();
		m_pendingChat.clear();
	}
	m_eventHub.signal(AS_BoardChange);
	m_eventHub.signal(AS_PlayerChange);
	m_eventHub.signal(AS_StateChange);
}

} // namespace tengen::app
