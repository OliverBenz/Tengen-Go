#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <variant>

namespace tengen {

struct Place {
	Coord coord;
};
struct Pass {};
using GameAction = std::variant<Place, Pass>;

struct GameMove {
	unsigned moveId;   //!< Move number.
	Player player;     //!< Player to make move.
	GameAction action; //!< Type of move performed.
};


} // namespace tengen
