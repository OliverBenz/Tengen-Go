#pragma once

#include "model/gameMove.hpp"
#include "model/gameResult.hpp"
#include "model/gameRules.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

// This file contains the 3 types of update data we get throughout a game.
//  1) Beginning of Game: We get a game config
//  2) During the Game:   We get   game deltas
//  3) After the Game:    We get a game result
namespace tengen {

//! How the game is to be played.
struct GameConfig {
	unsigned boardSize;
	GameRules rules;
	// TODO: Clock
};

//! Symbolises the game state change after one move.
struct GameDelta {
	GameMove move;               //!< The move that was performed.
	std::vector<Coord> captures; //!< Stones removed from the board. On suicide also the player's own.
};

//! How the game ended.
struct GameResult {
	std::optional<Player> winner; //!< Not set means no winner.
	EndReason reason;             //!< Reason for the game to end.
};

//! Board sizes a game can be played on, smallest first.
inline constexpr std::array<unsigned, 3> SUPPORTED_BOARD_SIZES{9u, 13u, 19u};

//! True if a game can be played on a board of this size.
constexpr bool isSupportedBoardSize(const unsigned size) {
	return std::ranges::find(SUPPORTED_BOARD_SIZES, size) != SUPPORTED_BOARD_SIZES.end();
}

} // namespace tengen
