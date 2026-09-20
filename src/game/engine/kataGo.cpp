#include "engine/kataGo.hpp"

#include "gtp.hpp"
#include "subProcess.hpp"

#include <filesystem>
#include <string>

namespace tengen::engine {
static constexpr const char* LOG_FILE = "katago.log"; //!< Takes over the engine's stderr.

static bool validConfig(const LaunchConfig& config) {
	std::error_code ec;
	return std::filesystem::exists(config.executable, ec) && std::filesystem::exists(config.model, ec) && std::filesystem::exists(config.config, ec) && std::filesystem::exists(config.modelHuman, ec);
}

KataGo::KataGo()
    : m_process{std::make_unique<SubProcess>()} {
}

KataGo::~KataGo() {
	stop();
}

bool KataGo::start(const LaunchConfig& config) {
	// Check valid config
	if (!validConfig(config)) {
		return false;
	}

	if (!m_process->start({config.executable,
	                       "gtp",
	                       "-model", config.model,
	                       "-human-model", config.modelHuman,
	                       "-config", config.config},
	                      LOG_FILE)) {
		return false;
	}

	// Forking succeeds even when the executable cannot be launched. Only an answer proves that we
	// are talking to a GTP engine.
	std::string response;
	if (!sendCommand(gtp::protocolVersion(), response)) {
		m_process->stop();
		return false;
	}
	return true;
}

void KataGo::stop() {
	// Ask the engine to shut down but do not wait for the answer: a genmove may still be blocking on
	// the pipe from another thread. Closing its stdin in stop() ends the engine either way.
	// This runs even when no process is attached yet, so that a start() still in flight is cancelled.
	m_process->sendLine(gtp::quit());
	m_process->stop();

	if (m_genmoveThread.joinable()) {
		m_genmoveThread.join();
	}
}

bool KataGo::startGame(const unsigned boardSize, const tengen::Player botColour) {
	m_boardSize = boardSize;
	m_botColour = botColour;

	std::string response;
	bool success = true;
	success &= sendCommand(gtp::boardSize(boardSize), response);
	success &= sendCommand(gtp::clearBoard(), response);
	success &= sendCommand(gtp::komi(7.5f), response);
	return success; // TODO: Take the komi from the game configuration.
}

bool KataGo::place(const tengen::Coord pos) {
	std::string response;
	return sendCommand(gtp::play(opponent(m_botColour), pos, m_boardSize), response);
}

bool KataGo::pass() {
	std::string response;
	return sendCommand(gtp::pass(opponent(m_botColour)), response);
}

bool KataGo::resign() {
	// GTP has no command for the opponent resigning. The game is simply over.
	return true;
}

void KataGo::genmove(std::function<void(bool, BotMove)> callback) {
	if (m_genmoveThread.joinable()) {
		m_genmoveThread.join(); // Retire the previous request. Only one is ever in flight.
	}

	m_genmoveThread = std::thread([this, callback = std::move(callback)] {
		// The engine plays the move on its own board, so it must not be relayed back with place().
		BotMove move{};
		std::string response;
		const bool ok = sendCommand(gtp::genmove(m_botColour), response) && gtp::parseMove(response, m_boardSize, move);
		callback(ok, move);
	});
}

bool KataGo::sendCommand(const std::string& command, std::string& response) {
	response.clear();

	std::string raw;
	if (!m_process->sendLine(command) || !m_process->readUntil(raw, gtp::responseEnd)) {
		return false;
	}

	return gtp::parseResponse(raw, response);
}

} // namespace tengen::engine
