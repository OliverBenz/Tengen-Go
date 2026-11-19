#include "botSession.hpp"

#include "core/gameEvent.hpp"
#include "logging.hpp"

#include <cassert>

namespace tengen::app {

BotSession::BotSession(const GameConfig& config, std::unique_ptr<engine::GtpEngine> botEngine, const bool playerPlaysAsBlack)
    : m_game(config), m_engine(std::move(botEngine)), m_botColour(playerPlaysAsBlack ? Player::White : Player::Black) {
	assert(m_engine);
	m_gameInfo.reset(config, GameStatus::Ready); // The bot is not up yet, so the board takes no moves.
	m_game.subscribeState(this);
	m_gameThread = std::thread([this] { m_game.run(); });

	// Bringing the engine up costs seconds, so it answers on its own thread like any other request.
	m_engine->registerListener(this);
	m_engine->start(config, m_botColour);
}

BotSession::~BotSession() {
	shutdown();
}

GameStatus BotSession::status() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getStatus();
}

Board BotSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getBoard();
}

Player BotSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getPlayer();
}

void BotSession::tryPlace(const unsigned x, const unsigned y) {
	if (m_status != Status::PlayerMove) {
		return; // Only the user's own turn is his to play.
	}

	// The user's own colour, never the one the position happens to be at: the Game refuses the move
	// while it is not his turn, so a second click cannot slip in as a move for the bot.
	m_game.pushEvent(PutStoneEvent{opponent(m_botColour), Coord{x, y}});
}

void BotSession::tryPass() {
	if (m_status != Status::PlayerMove) {
		return; // Only the user's own turn is his to play.
	}
	m_game.pushEvent(PassEvent{opponent(m_botColour)});
}

void BotSession::tryResign() {
	if (m_status == Status::Idle || m_status == Status::Finished) {
		return; // Nothing to resign from yet, or anymore.
	}

	m_game.pushEvent(ResignEvent{opponent(m_botColour)});
}

void BotSession::shutdown() {
	// Stopping the engine first releases a request that is still blocking on the pipe and retires the engine thread with it.
	// Once stop() returns, no answer can reach us anymore, so the Game below is ours alone to take down.
	m_engine->stop();

	m_game.pushEvent(ShutdownEvent{});
	if (m_gameThread.joinable()) {
		m_gameThread.join();
	}
	m_game.unsubscribeState(this);
}

void BotSession::subscribe(IAppSignalListener* listener, AppSignalMask signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void BotSession::unsubscribe(IAppSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}

void BotSession::onGameStart(const GameConfig& config) {
	bool started      = false;
	Player nextPlayer = Player::Black;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		started    = m_gameInfo.update(config);
		nextPlayer = m_gameInfo.getPlayer();
	}

	if (!started) {
		return;
	}

	m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StateChange);

	// The bot opens the game when it plays black.
	if (m_botColour == nextPlayer) {
		requestBotMove();
	} else {
		m_status = Status::PlayerMove;
	}
}

void BotSession::onGameDelta(const GameDelta& delta) {
	bool applied = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		applied = m_gameInfo.update(delta);

		// Hand the turn over while the game info is still held.
		if (applied) {
			m_status = m_gameInfo.getPlayer() == m_botColour ? Status::BotMove : Status::PlayerMove;
		}
	}

	if (!applied) {
		return;
	}

	if (std::holds_alternative<Place>(delta.move.action)) {
		m_eventHub.signal(AS_BoardChange | AS_PlayerChange | AS_StonePlaced);
	} else {
		m_eventHub.signal(AS_PlayerChange);
	}

	// The Game accepted the move, so it is part of the truth now: mirror it into the engine.
	// The bot's own moves are skipped; the engine already played them when it generated them.
	if (delta.move.player != m_botColour) {
		relayPlayerMove(delta);
	}

	if (m_status == Status::BotMove) {
		requestBotMove();
	}
}

void BotSession::relayPlayerMove(const GameDelta& delta) {
	// 'play' does not run a search, so the round trip is short enough to keep on the game thread.
	// Doing it here also keeps the engine's board in the order the Game accepted the moves in.
	const auto* place  = std::get_if<Place>(&delta.move.action);
	const bool relayed = place ? m_engine->place(place->coord) : m_engine->pass();

	// The engine either died or refused a move our rules accepted. Either way its board is behind the
	// Game's from here on, so every move it would still generate answers a position we are not in.
	// A move the Game hands us after shutdown has nowhere to go and is none of its doing.
	if (!relayed && m_engine->isRunning()) {
		endSession("[BotSession] Engine did not take the move. Its board no longer matches the game.");
	}
}

void BotSession::onGameEnd(const GameResult& result) {
	m_status = Status::Finished;

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

void BotSession::requestBotMove() {
	m_status = Status::Thinking;
	m_engine->genmove();
}

void BotSession::onEngineReady() {
	m_game.pushEvent(StartEvent{});
}

void BotSession::onMoveGenerated(const engine::BotMove& move) {
	if (m_status == Status::Finished) {
		return;
	}

	// TODO: The Game drops rejected moves silently, so a move our ruleset disagrees with leaves the bot idle.
	if (const auto* place = std::get_if<Place>(&move)) {
		m_game.pushEvent(PutStoneEvent{m_botColour, place->coord});
	} else if (std::holds_alternative<Pass>(move)) {
		m_game.pushEvent(PassEvent{m_botColour});
	} else {
		m_game.pushEvent(ResignEvent{m_botColour});
	}
}

void BotSession::onEngineFailed() {
	// Idle means the engine never came up. Anything else means it owes a move it cannot make anymore.
	endSession(m_status == Status::Idle ? "[BotSession] Engine failed to start. The bot cannot answer."
	                                    : "[BotSession] Engine failed to produce a move.");
}

void BotSession::endSession(const std::string& reason) {
	Logger().Log(Logging::LogLevel::Error, reason);

	// The bot owes a move it can never make. Close the game instead of leaving the user in front of
	// a board that takes no more clicks and still reads as active.
	m_status = Status::Finished;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);

		// A running game is lost by the bot, like a player that disconnects. One that never started has no result.
		if (m_gameInfo.getStatus() == GameStatus::Active) {
			m_gameInfo.update(GameResult{opponent(m_botColour), EndReason::Forfeit});
		} else if (m_gameInfo.getStatus() != GameStatus::Done) {
			m_gameInfo.reset(m_gameInfo.getConfig(), GameStatus::Done);
		}
	}
	m_eventHub.signal(AS_StateChange);
}

} // namespace tengen::app
