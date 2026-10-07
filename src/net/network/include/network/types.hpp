#pragma once

#include "model/player.hpp"

#include <cstdint>
#include <optional>

namespace tengen::network {

using SessionId = std::uint32_t;

//! The role in the game.
enum class Seat : std::uint8_t {
	None,    //!< Just connected.
	Black,   //!< Plays for black.
	White,   //!< Plays for white.
	Observer //!< Only gets updated on board change.
};

inline constexpr bool isPlayer(const Seat seat) {
	return seat == Seat::Black || seat == Seat::White;
}

//! The player a seat plays for. Nullopt for seats that do not play.
inline constexpr std::optional<Player> toPlayer(const Seat seat) {
	switch (seat) {
	case Seat::Black:
		return Player::Black;
	case Seat::White:
		return Player::White;
	case Seat::None:
	case Seat::Observer:
		return std::nullopt;
	}
	return std::nullopt;
}

inline constexpr Seat toSeat(const Player player) {
	return player == Player::Black ? Seat::Black : Seat::White;
}

} // namespace tengen::network
