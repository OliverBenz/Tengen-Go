#include "core/gameState.hpp"
#include "core/moveChecker.hpp"
#include "zobristHash.hpp"

#include <format>
#include <stdexcept>
#include <utility>

namespace tengen {

//! \throws std::invalid_argument if there is no hash for the board size.
static std::unique_ptr<IZobristHash> makeHasher(const std::size_t boardSize) {
	switch (boardSize) {
	case 9u:
		return std::make_unique<ZobristHash<9u>>();
	case 13u:
		return std::make_unique<ZobristHash<13u>>();
	case 19u:
		return std::make_unique<ZobristHash<19u>>();
	default:
		throw std::invalid_argument(std::format("Unsupported board size {}. Supported are 9, 13 and 19.", boardSize));
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

GameState::GameState(const GameConfig& config)
    : m_config{config}, m_position{config.boardSize}, m_hasher{makeHasher(config.boardSize)}, m_history{config.rules.koRule} {
	m_history.record(m_position.hash, m_position.currentPlayer);
}

bool GameState::start() {
	return !std::exchange(m_started, true);
}

std::optional<std::vector<Coord>> GameState::place(const Player player, const Coord c) {
	if (!isActive() || player != m_position.currentPlayer) {
		return std::nullopt;
	}

	auto placement = playStone(m_position.board, player, c, m_config.rules.suicideLegal);
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
	if (!isActive() || player != m_position.currentPlayer) {
		return false;
	}

	m_position.pass();
	m_history.record(m_position.hash, m_position.currentPlayer);

	++m_consecutivePasses;
	if (m_consecutivePasses == 2) {
		// TODO: Count the board. Until then the result names no winner.
		m_result = GameResult{.winner = std::nullopt, .reason = EndReason::Counting};
	}
	return true;
}

bool GameState::resign(const Player player) {
	if (!isActive()) {
		return false;
	}
	m_result = GameResult{.winner = opponent(player), .reason = EndReason::Resignation};
	return true;
}

bool GameState::isActive() const {
	return m_started && !m_result;
}

const std::optional<GameResult>& GameState::result() const {
	return m_result;
}

const GameConfig& GameState::config() const {
	return m_config;
}

const GamePosition& GameState::position() const {
	return m_position;
}

} // namespace tengen
