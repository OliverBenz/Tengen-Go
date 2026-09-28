#include "core/gameState.hpp"
#include "core/moveChecker.hpp"
#include "zobristHash.hpp"

#include <cassert>
#include <utility>

namespace tengen {

GameState::GameState(const std::size_t boardSize, const GameRules& rules) : m_rules{rules}, m_position{boardSize} {
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

std::optional<std::vector<Coord>> GameState::place(const Player player, const Coord c) {
	if (!m_active || player != m_position.currentPlayer) {
		return std::nullopt;
	}

	GamePosition next{m_position.board.size()};
	std::vector<Coord> captures{};
	if (!isNextPositionLegal(m_position, player, c, *m_hasher, m_seenHashes, next, captures)) {
		return std::nullopt;
	}

	m_position = std::move(next);
	m_seenHashes.insert(m_position.hash);
	m_consecutivePasses = 0;
	return captures;
}

bool GameState::pass(const Player player) {
	if (!m_active || player != m_position.currentPlayer) {
		return false;
	}

	m_position.pass(*m_hasher);
	m_seenHashes.insert(m_position.hash);

	++m_consecutivePasses;
	if (m_consecutivePasses == 2) {
		m_active = false;
	}
	return true;
}

bool GameState::resign() {
	return std::exchange(m_active, false);
}

bool GameState::isActive() const {
	return m_active;
}

const GamePosition& GameState::position() const {
	return m_position;
}

} // namespace tengen
