#pragma once

#include "model/board.hpp"
#include "model/player.hpp"

#include <cstdint>

namespace tengen {

//! The current game position.
struct GamePosition {
	Board board;                         //!< Current board.
	Player currentPlayer{Player::Black}; //!< Current Player.
	uint64_t hash{0};                    //!< Zobrist hash of the board alone. The empty board hashes to 0.
	unsigned moveId{0};                  //!< Move number of game.

public:
	explicit GamePosition(std::size_t boardSize);

	void play(Board nextBoard, uint64_t nextHash); //!< Current player made a move that left this board.
	void pass();                                   //!< Current player passes his turn.
};

} // namespace tengen
