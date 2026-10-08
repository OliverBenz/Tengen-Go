#include "tengen/openSession.hpp"

#include "core/gameEvent.hpp"

namespace tengen::app {

OpenSession::OpenSession(const GameConfig& config) : m_game(config) {
	// The board takes no moves until the Game signals the start.
	m_gameInfo.reset(config, GameStatus::Ready);

	m_game.subscribeState(this);
	m_gameThread = std::thread([this] { m_game.run(); });

	// Open play has nobody to wait for, so the game starts right away.
	m_game.pushEvent(StartEvent{});
}

OpenSession::~OpenSession() {
	shutdown();
}

GameStatus OpenSession::status() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getStatus();
}
Board OpenSession::board() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getBoard();
}
Player OpenSession::currentPlayer() const {
	std::lock_guard<std::mutex> lock(m_stateMutex);
	return m_gameInfo.getPlayer();
}

void OpenSession::tryPlace(const unsigned x, const unsigned y) {
	m_game.pushEvent(PutStoneEvent{currentPlayer(), Coord{x, y}});
}
void OpenSession::tryPass() {
	m_game.pushEvent(PassEvent{currentPlayer()});
}
void OpenSession::tryResign() {
	m_game.pushEvent(ResignEvent{currentPlayer()});
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

void OpenSession::onGameStart(const GameConfig& config) {
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

void OpenSession::onGameDelta(const GameDelta& delta) {
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
		m_eventHub.signal(AS_StonePlaced);
		break;
	case GameAction::Pass:
		m_eventHub.signal(AS_PlayerChange);
		break;
	case GameAction::Resign:
		break;
	}
}

void OpenSession::onGameEnd(const GameResult& result) {
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

} // namespace tengen::app
