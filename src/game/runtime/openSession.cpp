#include "tengen/openSession.hpp"

#include "core/gameEvent.hpp"

namespace tengen::app {

OpenSession::OpenSession(const std::size_t boardSize) : m_game(boardSize) {
	m_position.init(boardSize);
	m_game.subscribeState(this);
	m_gameThread = std::thread([this] { m_game.run(); });
}

OpenSession::~OpenSession() {
	shutdown();
}

GameStatus OpenSession::status() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getStatus();
}
Board OpenSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getBoard();
}
Player OpenSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_position.getPlayer();
}

void OpenSession::tryPlace(const unsigned x, const unsigned y) {
	m_game.pushEvent(PutStoneEvent{currentPlayer(), Coord{x, y}});
}
void OpenSession::tryPass() {
	m_game.pushEvent(PassEvent{currentPlayer()});
}
void OpenSession::tryResign() {
	m_game.pushEvent(ResignEvent{});
}
void OpenSession::shutdown() {
	m_game.pushEvent(ShutdownEvent{});
	if (m_gameThread.joinable()) {
		m_gameThread.join();
	}
	m_game.unsubscribeState(this);
}

void OpenSession::subscribe(IAppSignalListener* listener, uint64_t signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void OpenSession::unsubscribe(IAppSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}

void OpenSession::onGameDelta(const GameDelta& delta) {
	GameStatus status         = GameStatus::Active;
	GameStatus previousStatus = GameStatus::Active;
	bool applied              = false;
	{
		std::lock_guard<std::mutex> lock(m_stateMutex);
		previousStatus = m_position.getStatus();
		applied        = m_position.apply(delta);
		status         = m_position.getStatus();
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
}

} // namespace tengen::app
