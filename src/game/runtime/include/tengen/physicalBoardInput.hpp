#pragma once

#include "model/board.hpp"
#include "model/coordinate.hpp"
#include "tengen/IGameSession.hpp"

#include <chrono>
#include <optional>

namespace tengen::app {

//! Lets the player move by putting a stone on a physical board: one more way of input, next to clicking the board on screen.
//!
//! The game stays the source of truth. What is seen on the physical board is never taken over: a stone is only offered
//! to the session as the player's move, exactly like a click, and the session decides. Whoever watches the board
//! (a camera, say) just reports what it sees and knows nothing of the game.
//!
//! A stone is offered once the physical board shows the game's position plus that one stone, of the player to move,
//! and has done so for a moment: a stone held over the board or a hand resting on it is no move.
//! \note GUI thread only, like all input to a session.
class PhysicalBoardInput {
public:
	explicit PhysicalBoardInput(IGameSession& game);

	//! Report the stones currently seen on the physical board, in the game's orientation. Call it for every observation.
	void onBoardSeen(const Board& physical);

private:
	struct Candidate {
		Coord point;
		std::chrono::steady_clock::time_point since; //!< When it was first seen.
		bool offered;                                //!< Once is enough: the game answers through its board.
	};

	IGameSession& m_game;
	std::optional<Candidate> m_candidate; //!< The stone that may become the player's move. None while the boards agree or differ by more than that.
};

} // namespace tengen::app
