#include "tengen/physicalBoardInput.hpp"

#include "boardComparison.hpp"

namespace tengen::app {

static constexpr std::chrono::milliseconds SETTLE_TIME{1000}; //!< How long a stone has to lie on its point before it is the player's move.

PhysicalBoardInput::PhysicalBoardInput(IGameSession& game)
    : m_game(game) {
}

void PhysicalBoardInput::onBoardSeen(const Board& physical) {
	const BoardComparison comparison = compareBoards(m_game.board(), physical, m_game.currentPlayer());
	if (comparison.result != BoardComparison::Result::StonePlaced) {
		m_candidate.reset(); // The boards agree, or the physical one has to be brought in line with the game first.
		return;
	}

	const auto now    = std::chrono::steady_clock::now();
	const Coord point = comparison.placed;
	if (!m_candidate || m_candidate->point.x != point.x || m_candidate->point.y != point.y) {
		m_candidate = Candidate{.point = point, .since = now, .offered = false};
		return;
	}

	if (!m_candidate->offered && now - m_candidate->since >= SETTLE_TIME) {
		m_candidate->offered = true; // A stone the game refuses stays refused until it is taken off the board again.
		m_game.tryPlace(point.x, point.y);
	}
}

} // namespace tengen::app
