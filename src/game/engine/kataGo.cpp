#include "engine/kataGo.hpp"

#include "gtp.hpp"
#include "subProcess.hpp"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace tengen::engine {

static constexpr char LOG_FILE[] = "katago.log"; //!< Takes over the engine's stderr.

static bool validConfig(const LaunchConfig& config) {
	std::error_code ec;
	return std::filesystem::exists(config.executable, ec) && std::filesystem::exists(config.model, ec) && std::filesystem::exists(config.config, ec) && std::filesystem::exists(config.modelHuman, ec);
}

class KataGo::Implementation {
public:
	Implementation() = default;

	bool registerListener(IEngineListener* listener);

	void start(const LaunchConfig& config, unsigned boardSize, tengen::Player botColour);
	void stop();
	bool isRunning() const;

	bool place(tengen::Coord pos);
	bool pass();
	bool resign();

	void genmove();

private:
	void post(std::function<void()> request); //!< Hand one request to the engine thread. Dropped once stopped.
	void engineLoop();                        //!< Engine thread: run the posted requests until stop().
	bool canNotify() const;                   //!< False once stop() silenced the answers.

private:
	bool launch(const LaunchConfig& config);
	bool setupGame();
	bool sendCommand(const std::string& command, std::string& response); //!< Send one GTP command and wait for its response.

private:
	SubProcess m_process;
	IEngineListener* m_listener{nullptr};
	tengen::Player m_botColour{tengen::Player::Black}; //!< The player takes the other one.
	unsigned m_boardSize{9u};

	std::atomic<bool> m_running{false}; //!< Engine thread running.
	std::thread m_engineThread;
	std::mutex m_requestMutex;
	std::condition_variable m_requestReady;
	std::function<void()> m_pendingRequest; //!< Request waiting to be picked up. At most one.
};

bool KataGo::Implementation::registerListener(IEngineListener* listener) {
	if (m_listener) {
		return false;
	}
	m_listener = listener;
	return true;
}

void KataGo::Implementation::start(const LaunchConfig& config, const unsigned boardSize, const tengen::Player botColour) {
	assert(!m_running); // Starting twice would strand the engine that is already up.
	m_boardSize = boardSize;
	m_botColour = botColour;

	// Bringing the engine up costs seconds, so it is a request like any other.
	m_running      = true;
	m_engineThread = std::thread([this] { engineLoop(); });
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

void KataGo::Implementation::stop() {
	// Under the lock: it ends the engine loop and silences answers dying on the pipes killed below.
	{
		std::lock_guard<std::mutex> lock(m_requestMutex);
		m_running = false;
	}
	m_requestReady.notify_one();

	// Do not wait for the answer: a request may still be blocking on the pipe.
	m_process.sendLine(gtp::quit());
	m_process.stop();

	if (m_engineThread.joinable()) {
		m_engineThread.join();
	}
}

bool KataGo::Implementation::isRunning() const {
	return m_running;
}

bool KataGo::Implementation::place(const tengen::Coord pos) {
	std::string response;
	return sendCommand(gtp::play(opponent(m_botColour), pos, m_boardSize), response);
}

bool KataGo::Implementation::pass() {
	std::string response;
	return sendCommand(gtp::pass(opponent(m_botColour)), response);
}

bool KataGo::Implementation::resign() {
	// GTP has no command for the opponent resigning. The game is simply over.
	return true;
}

void KataGo::Implementation::genmove() {
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

void KataGo::Implementation::post(std::function<void()> request) {
	{
		std::lock_guard<std::mutex> lock(m_requestMutex);
		if (!m_running) {
			return; // Stopped. No thread left to run it, and nothing to answer with.
		}

		assert(!m_pendingRequest); // One request in flight at a time.
		m_pendingRequest = std::move(request);
	}
	m_requestReady.notify_one();
}

void KataGo::Implementation::engineLoop() {
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

bool KataGo::Implementation::canNotify() const {
	return m_running && m_listener;
}

bool KataGo::Implementation::launch(const LaunchConfig& config) {
	if (!validConfig(config)) {
		return false;
	}

	if (!m_process.start({config.executable,
	                      "gtp",
	                      "-model", config.model,
	                      "-human-model", config.modelHuman,
	                      "-config", config.config},
	                     LOG_FILE)) {
		return false;
	}

	// Starting the process succeeds even when it is not an engine. Only an answer proves GTP.
	std::string response;
	if (!sendCommand(gtp::protocolVersion(), response)) {
		m_process.stop();
		return false;
	}
	return true;
}

bool KataGo::Implementation::setupGame() {
	std::string response;
	bool success = true;
	success &= sendCommand(gtp::boardSize(m_boardSize), response);
	success &= sendCommand(gtp::clearBoard(), response);
	success &= sendCommand(gtp::komi(7.5f), response);
	return success; // TODO: Take the komi from the game configuration.
}

bool KataGo::Implementation::sendCommand(const std::string& command, std::string& response) {
	response.clear();

	std::string raw;
	if (!m_process.sendLine(command) || !m_process.readUntil(raw, gtp::responseEnd)) {
		return false;
	}

	return gtp::parseResponse(raw, response);
}


KataGo::KataGo()
    : m_pimpl(std::make_unique<Implementation>()) {
}

KataGo::~KataGo() {
	stop();
}

bool KataGo::registerListener(IEngineListener* listener) {
	return m_pimpl->registerListener(listener);
}

void KataGo::start(const LaunchConfig& config, const unsigned boardSize, const tengen::Player botColour) {
	m_pimpl->start(config, boardSize, botColour);
}

void KataGo::stop() {
	m_pimpl->stop();
}

bool KataGo::isRunning() const {
	return m_pimpl->isRunning();
}

bool KataGo::place(const tengen::Coord pos) {
	return m_pimpl->place(pos);
}

bool KataGo::pass() {
	return m_pimpl->pass();
}

bool KataGo::resign() {
	return m_pimpl->resign();
}

void KataGo::genmove() {
	m_pimpl->genmove();
}

} // namespace tengen::engine
