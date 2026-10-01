#include "core/position.hpp"

#include <utility>

namespace tengen {

GamePosition::GamePosition(std::size_t boardSize) : board{boardSize} {
}

void GamePosition::play(Board nextBoard, const uint64_t nextHash) {
	board = std::move(nextBoard);
	hash  = nextHash;

	currentPlayer = opponent(currentPlayer);
	++moveId;
}

void GamePosition::pass() {
	currentPlayer = opponent(currentPlayer);
	++moveId;
}

} // namespace tengen
