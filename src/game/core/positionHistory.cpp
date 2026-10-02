#include "core/positionHistory.hpp"

#include <cassert>

namespace tengen {

static constexpr std::size_t slot(const Player toMove) {
	return toMove == Player::Black ? 0u : 1u;
}

PositionHistory::PositionHistory(const Ko rule) : m_rule{rule} {
}

bool PositionHistory::allows(const uint64_t boardHash, const Player toMove) const {
	switch (m_rule) {
	case Ko::Simple:
		// Simple Ko: We cannot return to the same board position again.
		return boardHash != m_previousBoard;
	case Ko::Situational:
		// Situational Superko: A board position cannot be repeated for the player.
		return !m_seenBoards[slot(toMove)].contains(boardHash);
	case Ko::Positional:
		// Positional Superko: A board position generally cannot be repeated. Not just per-player.
		return !m_seenBoards[slot(Player::Black)].contains(boardHash) && !m_seenBoards[slot(Player::White)].contains(boardHash);
	}

	assert(false);
	return false;
}

void PositionHistory::record(const uint64_t boardHash, const Player toMove) {
	m_previousBoard = m_currentBoard;
	m_currentBoard  = boardHash;
	m_seenBoards[slot(toMove)].insert(boardHash);
}

} // namespace tengen
