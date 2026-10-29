#include "tengen/sessionGameInfo.hpp"
#include "logging.hpp"

#include <algorithm>
#include <cassert>

namespace tengen::app {

void SessionGameInfo::reset(const GameConfig& config, const GameStatus status) {
	m_config = config;
	m_result = std::nullopt;
	m_status = status;
	m_moveId = 0u;
	m_player = Player::Black;
	m_board  = Board{config.boardSize};
}

bool SessionGameInfo::update(const GameConfig& config) {
	if (m_status == GameStatus::Active) {
		Logger().Log(Logging::LogLevel::Error, "Received game start while a game is active.");
		return false;
	}

	// TODO: Timer not yet implemented.
	reset(config, GameStatus::Active);
	return true;
}

bool SessionGameInfo::update(const GameDelta& delta) {
	if (!isDeltaApplicable(delta)) {
		return false;
	}

	m_moveId = delta.moveId;
	m_player = delta.nextPlayer;

	if (delta.action == GameAction::Place) {
		assert(delta.coord); // isDeltaApplicable() rejects a place without one.
		m_board.place(*delta.coord, toStone(delta.player));
		for (const auto c: delta.captures) {
			m_board.remove(c);
		}
	}
	return true;
}

bool SessionGameInfo::update(const GameResult& result) {
	if (m_status != GameStatus::Active) {
		Logger().Log(Logging::LogLevel::Error, "Received game end while no game is active.");
		return false;
	}

	m_result = result;
	m_status = GameStatus::Done;
	return true;
}

const GameConfig& SessionGameInfo::getConfig() const {
	return m_config;
}
const Board& SessionGameInfo::getBoard() const {
	return m_board;
}
GameStatus SessionGameInfo::getStatus() const {
	return m_status;
}
Player SessionGameInfo::getPlayer() const {
	return m_player;
}
std::optional<GameResult> SessionGameInfo::getResult() const {
	return m_result;
}

bool SessionGameInfo::isDeltaApplicable(const GameDelta& delta) const {
	// Check the game status
	if (m_status != GameStatus::Active) {
		Logger().Log(Logging::LogLevel::Error, "Received game update before game is active.");
		return false;
	}

	// Check the move order
	if (delta.moveId <= m_moveId) {
		Logger().Log(Logging::LogLevel::Error, "Game delta sent to client twice.");
		return false;
	} else if (delta.moveId > m_moveId + 1) {
		Logger().Log(Logging::LogLevel::Error, "Game delta missing updates; applying latest update only.");

		// TODO: Query missing move and apply update first.
		return false;
	}

	// Check place action is valid
	if (delta.action == GameAction::Place) {
		if (!delta.coord) {
			Logger().Log(Logging::LogLevel::Error, "Game delta missing place coordinate.");
			return false;
		}

		// Board does not check coordinates: every one from the wire has to be on it.
		const auto size    = m_board.size();
		const auto onBoard = [size](const Coord c) { return c.x < size && c.y < size; };
		if (!onBoard(*delta.coord) || !std::ranges::all_of(delta.captures, onBoard)) {
			Logger().Log(Logging::LogLevel::Error, "Game delta coordinate outside the board.");
			return false;
		}
	}

	return true;
}

} // namespace tengen::app
