#pragma once

#include "model/player.hpp"

#include <cstdint>
#include <optional>

namespace tengen::network {

//! Identifies a client on the application layer. (enum class for strict typing and well defined empty)
enum class SessionId : std::uint32_t {
	Empty = 0, //!< No session: before the welcome, or after it ended. Never handed out.
};

using ProtocolVersion = std::uint32_t;

//! Version of the wire protocol.
inline constexpr ProtocolVersion PROTOCOL_VERSION = 1u;

//! Why the server refused a client.
enum class RejectReason : std::uint8_t {
	ProtocolMismatch, //!< The client speaks another protocol version.
};

//! The role in the game.
enum class Seat : std::uint8_t {
	None,    //!< Connected, but no hello yet.
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
