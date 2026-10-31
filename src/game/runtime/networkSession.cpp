#include "tengen/networkSession.hpp"

#include "logging.hpp"
#include "model/gameConfig.hpp"
#include "model/gameDelta.hpp"
#include "model/gameResult.hpp"
#include "tengen/gameServer.hpp"

#include <cassert>
#include <format>
#include <stdexcept>

namespace tengen::app {


NetworkSession::NetworkSession() {
	m_network.registerHandler(this);
}
NetworkSession::~NetworkSession() {
	disconnect();
	if (m_serverStopper.joinable()) {
		m_serverStopper.join();
	}
}

void NetworkSession::subscribe(IAppSignalListener* listener, AppSignalMask signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void NetworkSession::unsubscribe(IAppSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}


void NetworkSession::connect(const std::string& hostIp) {
	disconnect();
	m_network.connect(hostIp);
}

bool NetworkSession::host(const GameConfig& config, const Player hostColour) {
	disconnect();

	// The last server has to free the port first. Only waits if it is still stopping.
	if (m_serverStopper.joinable()) {
		m_serverStopper.join();
	}

	// Creating server may throw on invalid game config.
	try {
		m_localServer = std::make_unique<GameServer>(config, hostColour);
	} catch (const std::invalid_argument& e) {
		Logger().Log(Logging::LogLevel::Error, std::format("[NetworkSession] Cannot host the game: {}", e.what()));
		return false;
	}
	if (!m_localServer->start()) {
		m_localServer.reset();
		return false;
	}

	// We connect first, so we take the seat the server hands to its first player.
	m_network.connect("127.0.0.1");
	return true;
}

void NetworkSession::stopLocalServer() {
	if (!m_localServer) {
		return;
	}
	assert(!m_serverStopper.joinable()); // host() waited for the last stop before it made this server.
	m_serverStopper = std::thread([server = std::move(m_localServer)] { server->stop(); });
}

void NetworkSession::disconnect() {
	// onDisconnected() resets the state.
	m_network.disconnect();
	stopLocalServer();
}

void NetworkSession::shutdown() {
	disconnect();
}

void NetworkSession::resetState(const GameStatus status) {
	static const GameConfig PLACEHOLDER_CONFIG{9u, fromRuleSet(RuleSet::Japanese)};

	std::lock_guard<std::mutex> lock(m_stateMutex);
	m_gameInfo.reset(PLACEHOLDER_CONFIG, status);
	m_chatInfo.reset();
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
std::vector<ChatMessage> NetworkSession::getChatSince(const unsigned messageId) const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_chatInfo.getSince(messageId);
}
SessionNetworkInfo NetworkSession::networkInfo() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_networkInfo;
}

void NetworkSession::onConnecting() {
	resetState(GameStatus::Ready);
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_networkInfo.setConnecting();
	}
	m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StateChange | AS_ConnectionChange);
}
void NetworkSession::onWelcome(const std::optional<Player> player) {
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_networkInfo.setConnected(player);
	}
	m_eventHub.signal(AS_ConnectionChange);
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
	m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StateChange);
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
		m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StonePlaced);
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
void NetworkSession::onChatMessage(const ChatMessage& chat) {
	bool appended = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		appended = m_chatInfo.update(chat);
	}
	if (appended) {
		m_eventHub.signal(AS_NewChat);
	}
}
void NetworkSession::onDisconnected(const network::DisconnectReason reason) {
	resetState(GameStatus::Idle);
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_networkInfo.setIdle(reason);
	}
	m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StateChange | AS_ConnectionChange);
}

} // namespace tengen::app
