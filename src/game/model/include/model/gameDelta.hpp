#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <optional>
#include <vector>

namespace tengen {

//! Type of move.
enum class GameAction { Place, Pass, Resign };

//! Symbolises the game state change after one move.
struct GameDelta {
	unsigned moveId;             //!< Move number.
	GameAction action;           //!< Move type.
	Player player;               //!< Player to make move.
	std::optional<Coord> coord;  //!< For place action: Coordinate of place.
	std::vector<Coord> captures; //!< Stones removed from the board. On suicide also the player's own.
	Player nextPlayer;           //!< Next player to make a move. In case we add handicap, penalties, etc.
};

} // namespace tengen
