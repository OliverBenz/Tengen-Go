#pragma once

#include "model/gameRules.hpp"

#include <cstddef>

namespace tengen {

//! How the game is to be played.
struct GameConfig {
	std::size_t boardSize;
	GameRules rules;
	// TODO: Clock, Handicap
};

} // namespace tengen
