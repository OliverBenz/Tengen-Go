#include "kataGo.hpp"

#include <cassert>
#include <filesystem>
#include <format>
#include <string>

static bool validConfig(const LaunchConfig& config) {
	return std::filesystem::exists(config.executable) && std::filesystem::exists(config.model) && std::filesystem::exists(config.config) && std::filesystem::exists(config.modelHuman);
}

namespace gtp {

//! Convert a player to the GTP colour argument.
static std::string colour(const tengen::Player player) {
	return player == tengen::Player::Black ? "b" : "w";
}

//! Convert a board coordinate to a GTP vertex.
//! \note GTP skips the letter 'I' in the columns and numbers the rows from the bottom up.
static std::string vertex(const tengen::Coord pos, const unsigned boardSize) {
	static constexpr char columns[] = "ABCDEFGHJKLMNOPQRST";
	assert(pos.x < std::size(columns) - 1 && pos.y < boardSize);
	return std::string(1, columns[pos.x]) + std::to_string(boardSize - pos.y);
}

static std::string play(const tengen::Player player, const tengen::Coord position, const unsigned boardSize) {
	return std::format("play {} {}", colour(player), vertex(position, boardSize));
}

static std::string pass(const tengen::Player player) {
	return std::format("play {} pass", colour(player));
}

static std::string boardSize(const unsigned boardSize) {
	return std::format("boardsize {}", boardSize);
}

static std::string clearBoard() {
	return "clear_board";
}

static std::string komi(const float komi) {
	return std::format("komi {}", komi);
}

static std::string quit() {
	return "quit";
}

} // namespace gtp


KataGo::~KataGo() {
	stop();
}

bool KataGo::start(const LaunchConfig& config) {
	// Check valid config
	if (!validConfig(config)) {
		return false;
	}

	return m_process.start({config.executable,
	                        "gtp",
	                        "-model", config.model,
	                        "-human-model", config.modelHuman,
	                        "-config", config.config});
}

void KataGo::stop() {
	// Request stop from KataGo before the pipes go away.
	std::string response;
	m_process.sendCommand(gtp::quit(), response);

	m_process.stop();
}

bool KataGo::startGame(const unsigned boardSize, const tengen::Player botColour) {
	m_boardSize = boardSize;
	m_botColour = botColour;

	std::string response;
	bool success = true;
	success &= m_process.sendCommand(gtp::boardSize(boardSize), response);
	success &= m_process.sendCommand(gtp::clearBoard(), response);
	success &= m_process.sendCommand(gtp::komi(7.5), response);
	return success; // TODO: Take the komi from the game configuration.
}

bool KataGo::place(const tengen::Coord pos) {
	std::string response;
	return m_process.sendCommand(gtp::play(opponent(m_botColour), pos, m_boardSize), response);
}

bool KataGo::pass() {
	std::string response;
	return m_process.sendCommand(gtp::pass(opponent(m_botColour)));
}

bool KataGo::resign() {
	// GTP has no command for the opponent resigning. The game is simply over.
	return true;
}
