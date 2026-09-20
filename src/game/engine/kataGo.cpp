#include "engine/kataGo.hpp"

#include "gtp.hpp"
#include "subProcess.hpp"

#include <cassert>
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

bool KataGo::registerListener(IEngineListener* listener) {
	if (m_listener) {
		return false;
	}
	m_listener = listener;
	return true;
}

void KataGo::start(const LaunchConfig& config, const unsigned boardSize, const tengen::Player botColour) {
	assert(!m_running); // Starting twice would strand the engine that is already up.
	m_boardSize = boardSize;
	m_botColour = botColour;

	// Bringing the engine up costs seconds, so it is a request like any other.
	m_running = true;
	m_worker  = std::thread([this] { workerLoop(); });
	post([this, config] {
		const bool ready = launch(config) && setupGame();
		if (!canNotify()) {
			return;
		}
		if (ready) {
			m_listener->onEngineReady();
		} else {
			m_listener->onEngineFailed();
		}
	});
}

void KataGo::stop() {
	// Clear this first: everything below kills the pipes, and nothing dying on them from here on is a
	// failure worth reporting. It also ends the worker loop.
	// Under the lock, so that a worker about to wait for work cannot miss it.
	{
		std::lock_guard<std::mutex> lock(m_requestMutex);
		m_running = false;
	}
	m_requestReady.notify_one();

	// Ask the engine to shut down but do not wait for the answer: a request may still be blocking on
	// the pipe from the worker thread. Closing its stdin in stop() ends the engine either way.
	// This runs even when no process is attached yet, so that a start() still in flight is cancelled.
	m_process->sendLine(gtp::quit());
	m_process->stop();

	if (m_worker.joinable()) {
		m_worker.join();
	}
}

bool KataGo::isRunning() const {
	return m_running;
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

void KataGo::genmove() {
	post([this] {
		// The engine plays the move on its own board, so it must not be relayed back with place().
		BotMove move{};
		std::string response;
		const bool ok = sendCommand(gtp::genmove(m_botColour), response) && gtp::parseMove(response, m_boardSize, move);

		if (!canNotify()) {
			return;
		}
		if (ok) {
			m_listener->onMoveGenerated(move);
		} else {
			m_listener->onEngineFailed();
		}
	});
}

void KataGo::post(std::function<void()> request) {
	{
		std::lock_guard<std::mutex> lock(m_requestMutex);
		if (!m_running) {
			return; // Stopped. There is no worker left to run the request, and nothing to answer with.
		}

		// The worker takes a request out of the slot before running it, so an occupied slot means two
		// were posted without the first ever being picked up. A post from a callback is not that.
		assert(!m_pendingRequest);
		m_pendingRequest = std::move(request);
	}
	m_requestReady.notify_one();
}

void KataGo::workerLoop() {
	while (true) {
		std::function<void()> request;
		{
			std::unique_lock<std::mutex> lock(m_requestMutex);
			m_requestReady.wait(lock, [this] { return m_pendingRequest || !m_running; });
			if (!m_running) {
				return;
			}
			request          = std::move(m_pendingRequest);
			m_pendingRequest = nullptr;
		}

		// Outside the lock: the request blocks on the engine, and its answer may post the next one.
		request();
	}
}

bool KataGo::canNotify() const {
	return m_running && m_listener;
}

bool KataGo::launch(const LaunchConfig& config) {
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

bool KataGo::setupGame() {
	std::string response;
	bool success = true;
	success &= sendCommand(gtp::boardSize(m_boardSize), response);
	success &= sendCommand(gtp::clearBoard(), response);
	success &= sendCommand(gtp::komi(7.5f), response);
	return success; // TODO: Take the komi from the game configuration.
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
