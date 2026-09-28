#include "core/game.hpp"
#include "core/moveChecker.hpp"
#include "zobristHash.hpp"

namespace tengen {

Game::Game(const std::size_t boardSize, const GameRules& rules) : m_rules{rules}, m_position{boardSize} {
	switch (m_position.board.size()) {
	case 9u:
		m_hasher = std::make_unique<ZobristHash<9u>>();
		break;
	case 13u:
		m_hasher = std::make_unique<ZobristHash<13u>>();
		break;
	case 19u:
		m_hasher = std::make_unique<ZobristHash<19u>>();
		break;
	default:
		assert(false);
		break;
	}
	m_seenHashes.insert(m_position.hash);
}

void Game::pushEvent(GameEvent event) {
	m_eventQueue.Push(event);
}

void Game::run() {
	// Blocking loop: intended to live on its own thread.
	m_active = true;

	while (m_active) {
		const auto event = m_eventQueue.Pop();
		std::visit([&](auto&& ev) { handleEvent(ev); }, event);
	}
}

bool Game::isActive() const {
	return m_active;
}

std::size_t Game::boardSize() const {
	return m_position.board.size();
}

void Game::handleEvent(const PutStoneEvent& event) {
	assert(m_hasher);

	if (event.player != m_position.currentPlayer) {
		return;
	}

	GamePosition next{m_position.board.size()};
	std::vector<Coord> captures{};
	if (isNextPositionLegal(m_position, m_position.currentPlayer, event.c, *m_hasher, m_seenHashes, next, captures)) {
		m_consecutivePasses = 0;

		m_position = std::move(next);
		m_seenHashes.insert(m_position.hash);

		m_eventHub.signal(GS_BoardChange);
		m_eventHub.signal(GS_PlayerChange);
		m_eventHub.signalDelta(GameDelta{
		        .moveId     = m_position.moveId,
		        .action     = GameAction::Place,
		        .player     = event.player,
		        .coord      = event.c,
		        .captures   = captures,
		        .nextPlayer = m_position.currentPlayer,
		        .gameActive = m_active,
		});
	}
}

void Game::handleEvent(const PassEvent& event) {
	assert(m_hasher);

	if (event.player != m_position.currentPlayer) {
		return;
	}

	++m_consecutivePasses;
	if (m_consecutivePasses == 2) {
		m_active = false;
		m_eventHub.signalDelta(GameDelta{
		        .moveId     = m_position.moveId + 1,
		        .action     = GameAction::Pass,
		        .player     = event.player,
		        .coord      = std::nullopt,
		        .captures   = {},
		        .nextPlayer = opponent(event.player),
		        .gameActive = m_active,
		});
		m_eventHub.signal(GS_StateChange);
		return;
	}

	m_position.pass(*m_hasher);
	m_seenHashes.insert(m_position.hash);

	m_eventHub.signal(GS_PlayerChange);
	m_eventHub.signalDelta(GameDelta{
	        .moveId     = m_position.moveId,
	        .action     = GameAction::Pass,
	        .player     = event.player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = m_position.currentPlayer,
	        .gameActive = m_active,
	});
}

void Game::handleEvent(const ResignEvent&) {
	m_active = false;

	m_eventHub.signal(GS_StateChange);
	m_eventHub.signalDelta(GameDelta{
	        .moveId     = m_position.moveId + 1,
	        .action     = GameAction::Resign,
	        .player     = m_position.currentPlayer,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = opponent(m_position.currentPlayer),
	        .gameActive = m_active,
	});
}

void Game::handleEvent(const ShutdownEvent&) {
	m_active = false;
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
