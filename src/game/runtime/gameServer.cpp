#include "tengen/gameServer.hpp"

#include "core/game.hpp"
#include "logging.hpp"

#include <cassert>
#include <format>

namespace tengen::app {

static constexpr char LOG_REC_PUT[]    = "[GameServer] Received Event 'Put'    from player {} at ({}, {}).";
static constexpr char LOG_REC_PASS[]   = "[GameServer] Received Event 'Pass'   from Player {}.";
static constexpr char LOG_REC_RESIGN[] = "[GameServer] Received Event 'Resign' from Player {}.";

GameServer::GameServer(const GameConfig& config, const Player firstPlayer) : m_game(config) {
	m_server.setFirstSeat(firstPlayer == Player::Black ? network::Seat::Black : network::Seat::White);
}

GameServer::~GameServer() {
	stop();
}

void GameServer::start() {
	if (!m_server.registerHandler(this)) {
		Logger().Log(Logging::LogLevel::Warning, "[GameServer] Server handler already registered. Start ignored.");
		return;
	}
	m_game.subscribeState(this);

	m_gameThread = std::thread([this] { m_game.run(); }); // Start message loop but don't send the game start event yet.
	m_server.start();
}

void GameServer::stop() {
	if (m_gameThread.joinable()) {
		m_game.pushEvent(ShutdownEvent{});
	}

	m_server.stop();
	m_game.unsubscribeState(this);

	if (m_gameThread.joinable()) {
		m_gameThread.join();
	}
	m_seated.clear();
}

void GameServer::onPlayerJoined(const Player player) {
	Logger().Log(Logging::LogLevel::Info, std::format("[GameServer] Player {} joined.", toString(player)));

	// TODO: Handle reconnect. A returning player takes the seat back, and the Game ignores the second start.
	m_seated.insert(player);
	if (m_seated.size() == 2) {
		m_game.pushEvent(StartEvent{});
	}
}

void GameServer::onPlayerLeft(const Player player) {
	// TODO: Not handled for now. No timing in game.
	Logger().Log(Logging::LogLevel::Info, std::format("[GameServer] Player {} left.", toString(player)));

	m_seated.erase(player);
}

void GameServer::onPlace(const Player player, const Coord c) {
	Logger().Log(Logging::LogLevel::Info, std::format(LOG_REC_PUT, static_cast<int>(player), c.x, c.y));

	m_game.pushEvent(PutStoneEvent{player, c});
}

void GameServer::onPass(const Player player) {
	Logger().Log(Logging::LogLevel::Info, std::format(LOG_REC_PASS, static_cast<int>(player)));

	m_game.pushEvent(PassEvent{player});
}

void GameServer::onResign(const Player player) {
	Logger().Log(Logging::LogLevel::Info, std::format(LOG_REC_RESIGN, static_cast<int>(player)));

	m_game.pushEvent(ResignEvent{player});
}

void GameServer::onChat(const Player player, const std::string& message) {
	m_chatHistory.emplace_back(ChatEntry{player, message});
	m_server.broadcast(network::ServerChat{player, static_cast<unsigned>(m_chatHistory.size()), message});
}

void GameServer::onGameStart(const GameConfig& config) {
	m_server.broadcast(network::ServerGameStart{config});
}

void GameServer::onGameDelta(const GameDelta& delta) {
	m_server.broadcast(network::ServerGameDelta{delta});
}

void GameServer::onGameEnd(const GameResult& result) {
	m_server.broadcast(network::ServerGameEnd{result});
}

} // namespace tengen::app
