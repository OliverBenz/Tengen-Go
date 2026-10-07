#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <cstdint>
#include <variant>

namespace tengen {

struct StartEvent {};
struct PutStoneEvent {
	Player player;
	Coord c;
};
struct PassEvent {
	Player player;
};
struct ResignEvent {
	Player player;
};
struct ShutdownEvent {};
using GameEvent = std::variant<StartEvent, PutStoneEvent, PassEvent, ResignEvent, ShutdownEvent>;


//! Types of signals.
enum GameSignal : std::uint64_t {
	GS_None         = 0,
	GS_BoardChange  = 1 << 0, //!< Board was modified.
	GS_PlayerChange = 1 << 1, //!< Active player changed.
	GS_StateChange  = 1 << 2, //!< Game state changed. Started or finished.
};

} // namespace tengen
