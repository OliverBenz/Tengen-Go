#include "tengen/position.hpp"
#include "logging.hpp"

#include <cassert>

namespace tengen::app {

void Position::reset(const std::size_t boardSize) {
	m_moveId = 0u;
	m_status = GameStatus::Idle;
	m_player = Player::Black;
	m_board  = Board{boardSize};
}

bool Position::init(const std::size_t boardSize) {
	if (m_status == GameStatus::Active) {
		return false;
	}

	// TODO: Komi and timer not yet implemented.
	m_moveId = 0u;
	m_status = GameStatus::Active;
	m_player = Player::Black;
	m_board  = Board{boardSize};
	return true;
}

bool Position::apply(const GameDelta& delta) {
	if (!isDeltaApplicable(delta)) {
		return false;
	}

	m_moveId = delta.moveId;
	m_status = delta.gameActive ? GameStatus::Active : GameStatus::Done;
	m_player = delta.nextPlayer;

	if (delta.action == GameAction::Place) {
		if (delta.coord) {
			m_board.place(Coord{delta.coord->x, delta.coord->y}, toStone(delta.player));
			for (const auto c: delta.captures) {
				m_board.remove(c);
			}
		} else {
			Logger().Log(Logging::LogLevel::Warning, "Game delta missing place coordinate; skipping board update.");
		}
	}
	return true;
}

void Position::setStatus(GameStatus status) {
	m_status = status;
}


const Board& Position::getBoard() const {
	return m_board;
}
GameStatus Position::getStatus() const {
	return m_status;
}
Player Position::getPlayer() const {
	return m_player;
}

bool Position::isDeltaApplicable(const GameDelta& delta) {
	// No gamestate updates before game is active (received game configuration).
	if (m_status != GameStatus::Active) {
		Logger().Log(Logging::LogLevel::Error, "Received game update before game is active.");
		return false;
	}

	// Game delta for the proper move.
	if (delta.moveId <= m_moveId) {
		Logger().Log(Logging::LogLevel::Error, "Game delta sent to client twice.");
		return false;
	} else if (delta.moveId > m_moveId + 1) {
		Logger().Log(Logging::LogLevel::Error, "Game delta missing updates; applying latest update only.");

		// TODO: Query missing move and apply update first.
		return false;
	}

	return true;
}

} // namespace tengen::app
