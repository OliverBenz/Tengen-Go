#include "core/game.hpp"

namespace tengen {

Game::Game(const GameConfig& config) : m_state{config} {
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

void Game::handleEvent(const StartEvent&) {
	if (!m_state.start()) {
		return;
	}

	m_eventHub.signal(GS_StateChange);
	m_eventHub.signalStart(m_state.config());
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
	};

	// Second consecutive passes can end the game.
	if (m_state.isActive()) {
		m_eventHub.signal(GS_PlayerChange);
		m_eventHub.signalDelta(delta);
	} else {
		m_eventHub.signal(GS_StateChange);
		m_eventHub.signalDelta(delta);
		m_eventHub.signalEnd(*m_state.result());
	}
}

void Game::handleEvent(const ResignEvent& event) {
	if (!m_state.resign(event.player)) {
		return;
	}

	m_eventHub.signal(GS_StateChange);
	m_eventHub.signalDelta(GameDelta{
	        .moveId     = m_state.position().moveId + 1,
	        .action     = GameAction::Resign,
	        .player     = event.player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = opponent(event.player),
	});
	m_eventHub.signalEnd(*m_state.result());
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
