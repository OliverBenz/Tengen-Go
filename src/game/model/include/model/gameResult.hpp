#pragma once

#include <cstdint>

namespace tengen {

enum class EndReason : std::uint8_t {
	Resignation, //!< Player resigned.
	Counting,    //!< Both players passed and the board was counted.
	Timeout,     //!< Player ran out of time.
	Forfeit      //!< Illegal move under strict rules or disconnect.
};

} // namespace tengen
