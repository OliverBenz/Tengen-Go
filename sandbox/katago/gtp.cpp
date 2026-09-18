#include "gtp.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <format>
#include <string_view>

namespace gtp {

static constexpr std::string_view columns = "ABCDEFGHJKLMNOPQRST"; //!< Column letters. GTP skips 'I' so it cannot be confused with 'J'.

//! Convert a player to the GTP colour argument.
static std::string colour(const tengen::Player player) {
	return player == tengen::Player::Black ? "b" : "w";
}

//! Convert a board coordinate to a GTP vertex.
//! \note GTP skips the letter 'I' in the columns and numbers the rows from the bottom up.
static std::string vertex(const tengen::Coord pos, const unsigned boardSize) {
	assert(boardSize <= columns.size() && pos.x < boardSize && pos.y < boardSize);
	return std::string(1, columns[pos.x]) + std::to_string(boardSize - pos.y);
}

std::string play(const tengen::Player player, const tengen::Coord position, const unsigned boardSize) {
	return std::format("play {} {}", colour(player), vertex(position, boardSize));
}

std::string pass(const tengen::Player player) {
	return std::format("play {} pass", colour(player));
}

std::string genmove(const tengen::Player player) {
	return std::format("genmove {}", colour(player));
}

std::string boardSize(const unsigned boardSize) {
	return std::format("boardsize {}", boardSize);
}

std::string clearBoard() {
	return "clear_board";
}

std::string komi(const float komi) {
	return std::format("komi {}", komi);
}

std::string protocolVersion() {
	return "protocol_version";
}

std::string quit() {
	return "quit";
}

//! Drop the leading and trailing whitespace.
static std::string_view trim(const std::string_view text) {
	constexpr std::string_view whitespace = " \t\r\n";

	const auto first = text.find_first_not_of(whitespace);
	if (first == std::string_view::npos) {
		return {};
	}
	return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

//! Compare two keywords ignoring their case. The engine is free to answer in any case.
static bool equalsIgnoreCase(const std::string_view lhs, const std::string_view rhs) {
	const auto sameLetter = [](const char a, const char b) {
		return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
	};
	return std::ranges::equal(lhs, rhs, sameLetter);
}

bool parseResponse(const std::string& raw, std::string& value) {
	const std::string_view body = trim(raw);
	if (body.empty()) {
		return false;
	}

	const bool success = body.front() == '=';
	if (!success && body.front() != '?') {
		return false; // Not a GTP response at all.
	}

	// The status may carry the id of the command it answers. We never send one, so this is defensive.
	std::size_t valueStart = 1u;
	while (valueStart < body.size() && std::isdigit(static_cast<unsigned char>(body[valueStart]))) {
		++valueStart;
	}

	value = std::string(trim(body.substr(valueStart)));
	return success;
}

//! Convert a GTP column letter to a board column.
static bool column(const char letter, unsigned& x) {
	const auto index = columns.find(static_cast<char>(std::toupper(static_cast<unsigned char>(letter))));
	if (index == std::string_view::npos) {
		return false;
	}

	x = static_cast<unsigned>(index);
	return true;
}

//! Convert a GTP row number to a board row. Rows are numbered from the bottom up, we count from the top down.
static bool row(const std::string_view number, const unsigned boardSize, unsigned& y) {
	unsigned value               = 0u;
	const auto [parseEnd, error] = std::from_chars(number.data(), number.data() + number.size(), value);
	if (error != std::errc{} || parseEnd != number.data() + number.size()) {
		return false;
	}
	if (value < 1u || value > boardSize) {
		return false;
	}

	y = boardSize - value;
	return true;
}

bool parseMove(const std::string& value, const unsigned boardSize, BotMove& move) {
	if (equalsIgnoreCase(value, "pass")) {
		move = BotMove{.action = MoveAction::Pass};
		return true;
	}
	if (equalsIgnoreCase(value, "resign")) {
		move = BotMove{.action = MoveAction::Resign};
		return true;
	}

	// Anything else is a vertex: the column letter followed by the row number.
	if (value.size() < 2u) {
		return false;
	}

	tengen::Coord position{};
	if (!column(value.front(), position.x) || position.x >= boardSize) {
		return false;
	}
	if (!row(std::string_view(value).substr(1u), boardSize, position.y)) {
		return false;
	}

	move = BotMove{.action = MoveAction::Place, .coord = position};
	return true;
}

} // namespace gtp
