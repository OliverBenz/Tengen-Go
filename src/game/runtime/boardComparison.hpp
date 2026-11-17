#pragma once

#include "model/board.hpp"
#include "model/coordinate.hpp"
#include "model/player.hpp"

namespace tengen::app {

//! How the physical board relates to the board of the game.
struct BoardComparison {
	enum class Result {
		Same,        //!< Both hold the same stones.
		StonePlaced, //!< The physical board holds exactly one stone more: of the player to move, on a free point.
		Differs,     //!< Anything else. Nothing to tell the game.
	};

	Result result;
	Coord placed; //!< Only for Result::StonePlaced.
};

//! Compare the physical board, as it is seen, with the board of the game, to find the stone the player to move has placed.
//! Deliberately strict: captured stones still lying on the board, a hand over it, a stone of the wrong colour or
//! several new stones all make the boards differ, so nothing reaches the game until they agree again.
//! \param [in] game     The board of the game.
//! \param [in] physical The physical board as seen, in the orientation of game.
//! \param [in] toMove   The player whose turn it is.
//! \note Knows no rules: a stone the game would refuse (ko, suicide) is reported as placed all the same.
BoardComparison compareBoards(const Board& game, const Board& physical, Player toMove);

} // namespace tengen::app
