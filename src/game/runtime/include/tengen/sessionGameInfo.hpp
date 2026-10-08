#pragma once

#include "model/board.hpp"
#include "model/gameConfig.hpp"
#include "model/gameDelta.hpp"
#include "model/gameResult.hpp"
#include "model/gameStatus.hpp"
#include "model/player.hpp"
#include <optional>

namespace tengen::app {

//! The game as a session knows it. Only changes through the signals the Game reports. The Game stays the source of truth.
//! The game sends signals via the IGameStateListener interface. You can 'update' this class with the sent data to rebuild the game info locally per session.
class SessionGameInfo {
public:
	SessionGameInfo() = default;

	void reset(const GameConfig& config, GameStatus status); //!< Clear back to a fresh game of this config.

	bool update(const GameConfig& config); //!< Game started. Returns false if a game is already active.
	bool update(const GameDelta& delta);   //!< Move accepted. Returns false if the delta does not follow the last one.
	bool update(const GameResult& result); //!< Game ended. Returns false if no game is active.

	const GameConfig& getConfig() const;         //!< Get the game configuration.
	const Board& getBoard() const;               //!< Get the current board setup.
	GameStatus getStatus() const;                //!< Get the current status of the game.
	Player getPlayer() const;                    //!< Get the current player to make a move.
	std::optional<GameResult> getResult() const; //!< Only set if the game has ended.

private:
	bool isDeltaApplicable(const GameDelta& delta) const; //!< Check if the delta is ok to use for the update.

private:
	GameConfig m_config{9u, fromRuleSet(RuleSet::Japanese)}; //!< How the current game is played.
	std::optional<GameResult> m_result{std::nullopt};        //!< How the game ended.
	GameStatus m_status{GameStatus::Idle};                   //!< Current status of the game.

	unsigned m_moveId{0};           //!< Last move id in game.
	Player m_player{Player::Black}; //!< Player to move.
	Board m_board{9u};              //!< Current board.
};

} // namespace tengen::app
