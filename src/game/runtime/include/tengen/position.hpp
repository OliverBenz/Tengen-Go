#pragma once

#include "core/gameEvent.hpp"
#include "model/board.hpp"
#include "model/gameStatus.hpp"
#include "model/player.hpp"

namespace tengen::app {

class Position {
public:
	Position() = default;

	void reset(std::size_t boardSize); //!< Reset the position to some default data.

	// TODO: This init will become GameConfig onece we handle Rulesets, clock type, etc.
	bool init(const std::size_t boardSize); //!< Initialize the given position. Returns true if it changed state.
	bool apply(const GameDelta& delta);     //!< Apply a delta to the current position if ok.
	void setStatus(GameStatus status);      //!< Update the status.

	const Board& getBoard() const;
	GameStatus getStatus() const;
	Player getPlayer() const;

private:
	bool isDeltaApplicable(const GameDelta& delta); //!< Check if the delta is ok to use for the position update.

private:
	unsigned m_moveId{0};                  //!< Last move id in game.
	GameStatus m_status{GameStatus::Idle}; //!< Current status of the game.
	Player m_player{Player::Black};
	Board m_board{9u};
};

} // namespace tengen::app
