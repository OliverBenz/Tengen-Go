#pragma once

#include "botMove.hpp"
#include "model/player.hpp"

#include <string>
#include <string_view>

//! Commands and responses of the Go Text Protocol.
namespace tengen::engine::gtp {

inline constexpr std::string_view responseEnd = "\n\n"; //!< Every response is terminated by an empty line.


std::string play(const tengen::Player player, const tengen::Coord position, const unsigned boardSize);
std::string pass(const tengen::Player player);
std::string genmove(const tengen::Player player);
std::string boardSize(const unsigned boardSize);
std::string clearBoard();
std::string komi(const float komi);
std::string protocolVersion();
std::string quit();


//! Split a raw response into its status and its value. Responses read "= value" when the command succeeded and "? reason" when it failed.
bool parseResponse(const std::string& raw, std::string& value);

//! Parse the answer of a genmove command. The engine replies with a vertex, "pass" or "resign".
bool parseMove(const std::string& value, const unsigned boardSize, BotMove& move);

} // namespace tengen::engine::gtp
