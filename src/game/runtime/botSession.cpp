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

	// Bringing the engine up costs seconds, so it runs on the engine thread like any other request.
	// The session stays idle until it is up: the status only opens the board once it answers.
	m_engineThread = std::thread([this, boardSize, engineConfig] {
		const bool ready = m_engine.start(engineConfig) && m_engine.startGame(boardSize, m_botColour);
		if (m_shuttingDown) {
			return; // Closed again before the engine was even up.
		}
		if (!ready) {
			endSession("[BotSession] Engine failed to start. The bot cannot answer.");
			return;
		}

		{
			std::lock_guard<std::mutex> lock(m_stateMutex);
			m_position.setStatus(GameStatus::Active);
		}
		m_eventHub.signal(AS_StateChange);

		// The bot opens the game when it plays black.
		if (m_botColour == Player::Black) {
			playBotMove();
		} else {
			m_status = Status::PlayerMove;
		}
	});
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
	// Tells the engine thread that a request dying on the pipe below is our doing, not a failure.
	m_shuttingDown = true;

	// Stopping the engine first releases a genmove that is still blocking on the pipe, and with it
	// the game thread should it be waiting on that request.
	m_engine.stop();

	m_game.pushEvent(ShutdownEvent{});
	if (m_gameThread.joinable()) {
		m_gameThread.join();
	}
	m_game.unsubscribeState(this);

	// The game thread is gone, so the request is ours alone to retire now.
	joinEngineThread();
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
	switch (delta.action) {
	case GameAction::Place:
		assert(delta.coord);
		if (!delta.coord) {
			Logger().Log(Logging::LogLevel::Warning, "[BotSession] Game delta missing place coordinate; engine not updated.");
			return;
		}
		m_engine.place(*delta.coord);
		break;
	case GameAction::Pass:
		m_engine.pass();
		break;
	case GameAction::Resign:
		m_engine.resign();
		break;
	}
}

void BotSession::requestBotMove() {
	if (m_shuttingDown) {
		return; // The engine is on its way out. There is nothing left to ask it.
	}
	joinEngineThread(); // Retire the previous request. Only one is ever in flight.

	m_engineThread = std::thread([this] { playBotMove(); });
}

void BotSession::playBotMove() {
	m_status = Status::Thinking;

	engine::BotMove move{};
	const bool answered = m_engine.genmove(move);
	if (m_shuttingDown) {
		return; // stop() pulled the pipe out from under the request, or the Game is already gone.
	}

	if (!answered) {
		endSession("[BotSession] Engine failed to produce a move.");
		return;
	}
	pushBotMove(move);
}

void BotSession::pushBotMove(const engine::BotMove& move) {
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

void BotSession::joinEngineThread() {
	if (m_engineThread.joinable()) {
		m_engineThread.join();
	}
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
