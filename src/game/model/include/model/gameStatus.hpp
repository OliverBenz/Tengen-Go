#pragma once

#include <cstdint>

namespace tengen {

enum class GameStatus : std::uint8_t {
	Idle,   //!< Nothing happening.
	Ready,  //!< Players ready to play.
	Active, //!< Game being played.
	Done    //!< Game over.
};

} // namespace tengen
