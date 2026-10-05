#pragma once

#include "core/IZobristHash.hpp"
#include "core/position.hpp"
#include "core/positionHistory.hpp"
#include "model/coordinate.hpp"
#include "model/gameConfig.hpp"
#include "model/gameResult.hpp"
#include "model/player.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace tengen {

//! Tracks the state of a single game. Registers the player moves according to the provided game rules.
class GameState {
public:
	//! Set up the game. Moves are refused until it is started.
	//! \throws std::invalid_argument if the board size is not 9, 13 or 19.
	explicit GameState(const GameConfig& config);
	bool start(); //!< Open the game for moves. Returns false if the game is already active.

	//! Place a stone for the player.
	//! \returns The stones removed from the board (on suicide also the player's own), or nullopt if nothing changed (game is not on, out of turn or illegal).
	std::optional<std::vector<Coord>> place(Player player, Coord c);

	//! Pass for the player. Two consecutive passes end the game.
	//! \returns False if the game is not on or out of turn.
	bool pass(Player player);

	//! The player resigns, ending the game. Allowed on either player's turn.
	//! \returns False if the game is not on.
	bool resign(Player player);

	bool isActive() const;                           //!< True from the start until the game ends by passes or resignation.
	const std::optional<GameResult>& result() const; //!< How the game ended. Empty until it ended.
	const GameConfig& config() const;                //!< How the game is played.
	const GamePosition& position() const;            //!< The current position.

private:
	bool m_started{false};              //!< Moves are only taken once started and until there is a result.
	std::optional<GameResult> m_result; //!< Set when the game ends.
	unsigned m_consecutivePasses{0};    //!< Two consecutive passes end the game.

	GameConfig m_config;                    //!< How the current game is played.
	GamePosition m_position;                //!< Stores the current game position.
	std::unique_ptr<IZobristHash> m_hasher; //!< Hash matching the board size.
	PositionHistory m_history;              //!< Applies the ko rule.
};

} // namespace tengen
