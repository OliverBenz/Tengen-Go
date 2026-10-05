#pragma once

#include "model/player.hpp"

#include <optional>

namespace tengen {

enum class EndReason {
	Resignation, //!< Player resigned.
	Counting,    //!< Both players passed and the board was counted.
	Timeout,     //!< Player ran out of time.
	Forfeit      //!< Illegal move under strict rules or disconnect.
};

//! How the game ended.
struct GameResult {
	std::optional<Player> winner; //!< Not set means no winner.
	EndReason reason;             //!< Reason for the game to end.
};

} // namespace tengen
