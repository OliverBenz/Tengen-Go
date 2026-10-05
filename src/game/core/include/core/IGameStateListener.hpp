#pragma once

#include "core/gameEvent.hpp"
#include "model/gameConfig.hpp"
#include "model/gameResult.hpp"

namespace tengen {

//! Receives the course of a game as data, in order: one start, one delta per accepted move, one end.
//! \note Subscribe before the StartEvent is handled. A listener that subscribes later never gets onGameStart() and with it the config.
class IGameStateListener {
public:
	virtual ~IGameStateListener()                      = default;
	virtual void onGameStart(const GameConfig& config) = 0; //!< Once, when the game started. Before any delta.
	virtual void onGameDelta(const GameDelta& delta)   = 0; //!< Once per accepted move.
	virtual void onGameEnd(const GameResult& result)   = 0; //!< Once, right after the delta of the move that ended the game.
};

} // namespace tengen
