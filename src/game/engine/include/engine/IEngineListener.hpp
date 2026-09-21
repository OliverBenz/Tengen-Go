#pragma once

#include "engine/botMove.hpp"

namespace tengen::engine {

//! Callback interface invoked on the engine's request thread.
//! \note Keep handlers lightweight.
class IEngineListener {
public:
	virtual ~IEngineListener() = default;

	virtual void onEngineReady()                      = 0;
	virtual void onMoveGenerated(const BotMove& move) = 0; //!< The engine already played the move on its own board.
	virtual void onEngineFailed()                     = 0;
};

} // namespace tengen::engine
