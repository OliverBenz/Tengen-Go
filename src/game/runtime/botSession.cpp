#include "tengen/botSession.hpp"

#include "core/gameEvent.hpp"
#include "logging.hpp"

#include <cassert>

namespace tengen::app {

BotSession::BotSession(const unsigned boardSize, const engine::LaunchConfig& engineConfig, const bool playerPlaysAsBlack)
    : m_game(boardSize), m_botColour(playerPlaysAsBlack ? Player::White : Player::Black) {
	m_position.init(boardSize);
	m_position.setStatus(GameStatus::Ready); // The bot is not up yet, so the board takes no moves.
	m_game.subscribeState(this);
	m_gameThread = std::thread([this] { m_game.run(); });

	// Bringing the engine up costs seconds, so it answers on its own thread like any other request.
	// The session stays idle until it is up: the status only opens the board once it answers.
	m_engine.registerListener(this);
	m_engine.start(engineConfig, boardSize, m_botColour);
}

BotSession::~BotSession() {
	shutdown();
}

GameStatus BotSession::status() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getStatus();
}

Board BotSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getBoard();
}

Player BotSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getPlayer();
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

	// TODO: ResignEvent names no player, so resigning while the bot thinks resigns in its name.
	m_game.pushEvent(ResignEvent{});
}

void BotSession::shutdown() {
	// Stopping the engine first releases a request that is still blocking on the pipe and retires the
	// engine thread with it. Once stop() returns, no answer can reach us anymore, so the Game below is
	// ours alone to take down.
	m_engine.stop();

	m_game.pushEvent(ShutdownEvent{});
	if (m_gameThread.joinable()) {
		m_gameThread.join();
	}
	m_game.unsubscribeState(this);
}

void BotSession::subscribe(IAppSignalListener* listener, uint64_t signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void BotSession::unsubscribe(IAppSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}

void BotSession::onGameDelta(const GameDelta& delta) {
	GameStatus status         = GameStatus::Active;
	GameStatus previousStatus = GameStatus::Active;
	bool applied              = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		previousStatus = m_position.getStatus();
		applied        = m_position.apply(delta);
		status         = m_position.getStatus();

		// Hand the turn over while the position is still held. Everything below may block, and for
		// as long as it does tryPlace() must not see a turn the Game has already moved past.
		if (applied) {
			if (status != GameStatus::Active) {
				m_status = Status::Finished;
			} else {
				m_status = delta.nextPlayer == m_botColour ? Status::BotMove : Status::PlayerMove;
			}
		}
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
	if (previousStatus != status) {
		m_eventHub.signal(AS_StateChange);
	}

	// The Game accepted the move, so it is part of the truth now: mirror it into the engine.
	// The bot's own moves are skipped; the engine already played them when it generated them.
	if (delta.player != m_botColour) {
		relayPlayerMove(delta);
	}

	if (m_status == Status::BotMove) {
		requestBotMove();
	}
}

void BotSession::relayPlayerMove(const GameDelta& delta) {
	// 'play' does not run a search, so the round trip is short enough to keep on the game thread.
	// Doing it here also keeps the engine's board in the order the Game accepted the moves in.
	// Only the player's own moves get here, so the engine is never thinking while we send.
	bool relayed = true;
	switch (delta.action) {
	case GameAction::Place:
		assert(delta.coord);
		if (!delta.coord) {
			Logger().Log(Logging::LogLevel::Warning, "[BotSession] Game delta missing place coordinate; engine not updated.");
			return;
		}
		relayed = m_engine.place(*delta.coord);
		break;
	case GameAction::Pass:
		relayed = m_engine.pass();
		break;
	case GameAction::Resign:
		relayed = m_engine.resign();
		break;
	}

	// The engine either died or refused a move our rules accepted. Either way its board is behind the
	// Game's from here on, so every move it would still generate answers a position we are not in.
	// A move the Game hands us after shutdown has nowhere to go and is none of its doing.
	if (!relayed && m_engine.isRunning()) {
		endSession("[BotSession] Engine did not take the move. Its board no longer matches the game.");
	}
}

void BotSession::requestBotMove() {
	m_status = Status::Thinking;
	m_engine.genmove();
}

void BotSession::onEngineReady() {
	Player nextPlayer = Player::Black;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		m_position.setStatus(GameStatus::Active);
		nextPlayer = m_position.getPlayer();
	}
	m_eventHub.signal(AS_StateChange);

	// The bot opens the game when it plays black.
	if (m_botColour == nextPlayer) {
		requestBotMove();
	} else {
		m_status = Status::PlayerMove;
	}
}

void BotSession::onMoveGenerated(const engine::BotMove& move) {
	// The Game validates the move like any other. It signals us back through onGameDelta() when it accepts.
	// TODO: The Game drops rejected moves silently, so a move our ruleset disagrees with leaves the bot idle.
	switch (move.action) {
	case engine::MoveAction::Place:
		m_game.pushEvent(PutStoneEvent{m_botColour, move.coord});
		break;
	case engine::MoveAction::Pass:
		m_game.pushEvent(PassEvent{m_botColour});
		break;
	case engine::MoveAction::Resign:
		m_game.pushEvent(ResignEvent{});
		break;
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
		m_position.setStatus(GameStatus::Done);
	}
	m_eventHub.signal(AS_StateChange);
}

} // namespace tengen::app
