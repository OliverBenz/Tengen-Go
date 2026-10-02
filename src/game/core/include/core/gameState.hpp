#pragma once

#include "core/IZobristHash.hpp"
#include "core/position.hpp"
#include "core/positionHistory.hpp"
#include "model/coordinate.hpp"
#include "model/gameRules.hpp"
#include "model/player.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace tengen {

//! Tracks the state of a single game. Registers the player moves according to the provided game rules.
class GameState {
public:
	GameState(std::size_t boardSize, const GameRules& rules);

	//! Place a stone for the player.
	//! \returns The stones removed from the board (on suicide also the player's own), or nullopt if nothing changed (game is over, out of turn or illegal).
	std::optional<std::vector<Coord>> place(Player player, Coord c);

	//! Pass for the player. Two consecutive passes end the game.
	//! \returns False if the game is over or out of turn.
	bool pass(Player player);

	// TODO: Take the resigning player and record the opponent as winner instead of reporting a draw.
	//! The player to move resigns, ending the game.
	//! \returns False if the game is already over.
	bool resign();

	bool isActive() const;                //!< False once the game ended by passes or resignation.
	const GamePosition& position() const; //!< The current position.

private:
	bool m_active{true};             //!< False once the game ended.
	unsigned m_consecutivePasses{0}; //!< Two consecutive passes end the game.

	GameRules m_rules;                      //!< The rules we use for our current game.
	GamePosition m_position;                //!< Stores the current game position.
	std::unique_ptr<IZobristHash> m_hasher; //!< Hash matching the board size.
	PositionHistory m_history;              //!< Applies the ko rule.
};

} // namespace tengen
