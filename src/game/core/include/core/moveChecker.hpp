#pragma once

#include "model/board.hpp"
#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace tengen {

//! The board after a stone was played and what it took off.
struct Placement {
	Board board;                     //!< Board after the move.
	std::vector<Coord> captured;     //!< Opponent stones removed.
	std::vector<Coord> selfCaptured; //!< Own stones removed. Only when suicide is legal.
};

//! Play a stone on the board: capture enemy groups without liberties first, then apply the suicide rule.
//! \returns nullopt if the point is off the board, occupied, or the move is an illegal suicide.
//! \note Board mechanics only. Turn order and ko depend on the game history and live in GameState.
std::optional<Placement> playStone(const Board& board, Player player, Coord c, bool suicideLegal);

//! Liberties of the group connected to c if the player placed a stone there. Zero if c holds an enemy stone.
std::size_t computeGroupLiberties(const Board& board, Coord c, Player player);

} // namespace tengen
