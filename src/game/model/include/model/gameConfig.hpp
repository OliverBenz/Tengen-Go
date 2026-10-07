#pragma once

#include "model/gameRules.hpp"

#include <algorithm>
#include <array>

namespace tengen {

//! How the game is to be played.
struct GameConfig {
	std::size_t boardSize;
	GameRules rules;
	// TODO: Clock, Handicap
};

//! Board sizes a game can be played on, smallest first.
inline constexpr std::array<std::size_t, 3> SUPPORTED_BOARD_SIZES{9u, 13u, 19u};

//! True if a game can be played on a board of this size.
constexpr bool isSupportedBoardSize(const std::size_t size) {
	return std::ranges::find(SUPPORTED_BOARD_SIZES, size) != SUPPORTED_BOARD_SIZES.end();
}

} // namespace tengen
