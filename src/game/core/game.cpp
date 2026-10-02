#include "core/game.hpp"

namespace tengen {

Game::Game(const std::size_t boardSize, const GameRules& rules) : m_state{boardSize, rules} {
}

void Game::pushEvent(GameEvent event) {
	m_eventQueue.Push(event);
}

void Game::run() {
	// Blocking loop: intended to live on its own thread.
	m_running = true;

	while (m_running) {
		const auto event = m_eventQueue.Pop();
		std::visit([&](auto&& ev) { handleEvent(ev); }, event);
	}
}

std::size_t Game::boardSize() const {
	return m_state.position().board.size();
}

void Game::handleEvent(const PutStoneEvent& event) {
	const auto captures = m_state.place(event.player, event.c);
	if (!captures) {
		return;
	}

	m_eventHub.signal(GS_BoardChange);
	m_eventHub.signal(GS_PlayerChange);
	m_eventHub.signalDelta(GameDelta{
	        .moveId     = m_state.position().moveId,
	        .action     = GameAction::Place,
	        .player     = event.player,
	        .coord      = event.c,
	        .captures   = *captures,
	        .nextPlayer = m_state.position().currentPlayer,
	        .gameActive = m_state.isActive(),
	});
}

void Game::handleEvent(const PassEvent& event) {
	if (!m_state.pass(event.player)) {
		return;
	}

	const GameDelta delta{
	        .moveId     = m_state.position().moveId,
	        .action     = GameAction::Pass,
	        .player     = event.player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = m_state.position().currentPlayer,
	        .gameActive = m_state.isActive(),
	};

	// Second consecutive passes can end the game.
	if (m_state.isActive()) {
		m_eventHub.signal(GS_PlayerChange);
	} else {
		m_eventHub.signal(GS_StateChange);
	}
	m_eventHub.signalDelta(delta);
}

void Game::handleEvent(const ResignEvent&) {
	if (!m_state.resign()) {
		return;
	}

	const auto& position = m_state.position();
	m_eventHub.signal(GS_StateChange);
	m_eventHub.signalDelta(GameDelta{
	        .moveId     = position.moveId + 1,
	        .action     = GameAction::Resign,
	        .player     = position.currentPlayer,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = opponent(position.currentPlayer),
	        .gameActive = m_state.isActive(),
	});
}

void Game::handleEvent(const ShutdownEvent&) {
	m_running = false;
}

void Game::subscribeSignals(IGameSignalListener* listener, uint64_t signalMask) {
	m_eventHub.subscribe(listener, signalMask);
}

void Game::unsubscribeSignals(IGameSignalListener* listener) {
	m_eventHub.unsubscribe(listener);
}

void Game::subscribeState(IGameStateListener* listener) {
	m_eventHub.subscribe(listener);
}

void Game::unsubscribeState(IGameStateListener* listener) {
	m_eventHub.unsubscribe(listener);
}

} // namespace tengen
