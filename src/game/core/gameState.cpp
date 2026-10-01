#include "core/gameState.hpp"
#include "core/moveChecker.hpp"
#include "zobristHash.hpp"

#include <cassert>
#include <utility>

namespace tengen {

static std::unique_ptr<IZobristHash> makeHasher(const std::size_t boardSize) {
	switch (boardSize) {
	case 9u:
		return std::make_unique<ZobristHash<9u>>();
	case 13u:
		return std::make_unique<ZobristHash<13u>>();
	case 19u:
		return std::make_unique<ZobristHash<19u>>();
	default:
		assert(false);
		return nullptr;
	}
}

//! Board hash after the player placed a stone at c, updated from the hash before the move.
//! XOR toggles a stone in or out: add the placed stone, remove every captured stone.
//! On suicide, the removed own stones include c itself, so the placed stone cancels out again.
static uint64_t hashAfter(const Placement& placement, const uint64_t hash, IZobristHash& hasher, const Player player, const Coord c) {
	uint64_t next = hash ^ hasher.stone(c, player);
	for (const auto stone: placement.captured) {
		next ^= hasher.stone(stone, opponent(player));
	}
	for (const auto stone: placement.selfCaptured) {
		next ^= hasher.stone(stone, player);
	}
	return next;
}

GameState::GameState(const std::size_t boardSize, const GameRules& rules)
    : m_rules{rules}, m_position{boardSize}, m_hasher{makeHasher(boardSize)}, m_history{rules.koRule} {
	m_history.record(m_position.hash, m_position.currentPlayer);
}

std::optional<std::vector<Coord>> GameState::place(const Player player, const Coord c) {
	if (!m_active || player != m_position.currentPlayer) {
		return std::nullopt;
	}

	auto placement = playStone(m_position.board, player, c, m_rules.suicideLegal);
	if (!placement) {
		return std::nullopt;
	}

	const auto nextHash = hashAfter(*placement, m_position.hash, *m_hasher, player, c);
	if (!m_history.allows(nextHash, opponent(player))) {
		return std::nullopt;
	}

	m_position.play(std::move(placement->board), nextHash);
	m_history.record(m_position.hash, m_position.currentPlayer);
	m_consecutivePasses = 0;

	auto removed = std::move(placement->captured);
	removed.insert(removed.end(), placement->selfCaptured.begin(), placement->selfCaptured.end());
	return removed;
}

bool GameState::pass(const Player player) {
	if (!m_active || player != m_position.currentPlayer) {
		return false;
	}

	m_position.pass();
	m_history.record(m_position.hash, m_position.currentPlayer);

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
