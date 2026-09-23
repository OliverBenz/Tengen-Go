#include "engine/gtpEngine.hpp"

#include "gtp.hpp"
#include "subProcess.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace tengen::engine {

class GtpEngine::Implementation {
public:
	Implementation() = default;

	bool registerListener(IEngineListener* listener);

	void start(Launch command, unsigned boardSize, tengen::Player botColour);
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
	bool launch(const Launch& command);
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

bool GtpEngine::Implementation::registerListener(IEngineListener* listener) {
	if (m_listener) {
		return false;
	}
	m_listener = listener;
	return true;
}

void GtpEngine::Implementation::start(Launch command, const unsigned boardSize, const tengen::Player botColour) {
	assert(!m_running); // Starting twice would strand the engine that is already up.
	m_boardSize = boardSize;
	m_botColour = botColour;

	// Bringing an engine up can cost seconds, so it is a request like any other.
	m_running      = true;
	m_engineThread = std::thread([this] { engineLoop(); });
	post([this, command = std::move(command)] {
		const bool ready = launch(command) && setupGame();
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

void GtpEngine::Implementation::stop() {
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

bool GtpEngine::Implementation::isRunning() const {
	return m_running;
}

bool GtpEngine::Implementation::place(const tengen::Coord pos) {
	std::string response;
	return sendCommand(gtp::play(opponent(m_botColour), pos, m_boardSize), response);
}

bool GtpEngine::Implementation::pass() {
	std::string response;
	return sendCommand(gtp::pass(opponent(m_botColour)), response);
}

bool GtpEngine::Implementation::resign() {
	// GTP has no command for the opponent resigning. The game is simply over.
	return true;
}

void GtpEngine::Implementation::genmove() {
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

void GtpEngine::Implementation::post(std::function<void()> request) {
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

void GtpEngine::Implementation::engineLoop() {
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

bool GtpEngine::Implementation::canNotify() const {
	return m_running && m_listener;
}

bool GtpEngine::Implementation::launch(const Launch& command) {
	std::error_code ec;
	const auto exists = [&ec](const std::string& file) { return std::filesystem::exists(file, ec); };
	if (!std::ranges::all_of(command.requiredFiles, exists)) {
		return false;
	}

	if (!m_process.start(command.argv, command.logFile)) {
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

bool GtpEngine::Implementation::setupGame() {
	std::string response;
	bool success = true;
	success &= sendCommand(gtp::boardSize(m_boardSize), response);
	success &= sendCommand(gtp::clearBoard(), response);
	success &= sendCommand(gtp::komi(7.5f), response);
	return success; // TODO: Take the komi from the game configuration.
}

bool GtpEngine::Implementation::sendCommand(const std::string& command, std::string& response) {
	response.clear();

	std::string raw;
	if (!m_process.sendLine(command) || !m_process.readUntil(raw, gtp::responseEnd)) {
		return false;
	}

	return gtp::parseResponse(raw, response);
}


GtpEngine::GtpEngine()
    : m_pimpl(std::make_unique<Implementation>()) {
}

GtpEngine::~GtpEngine() {
	stop();
}

bool GtpEngine::registerListener(IEngineListener* listener) {
	return m_pimpl->registerListener(listener);
}

void GtpEngine::stop() {
	m_pimpl->stop();
}

bool GtpEngine::isRunning() const {
	return m_pimpl->isRunning();
}

bool GtpEngine::place(const tengen::Coord pos) {
	return m_pimpl->place(pos);
}

bool GtpEngine::pass() {
	return m_pimpl->pass();
}

bool GtpEngine::resign() {
	return m_pimpl->resign();
}

void GtpEngine::genmove() {
	m_pimpl->genmove();
}

void GtpEngine::launch(Launch command, const unsigned boardSize, const tengen::Player botColour) {
	m_pimpl->start(std::move(command), boardSize, botColour);
}

} // namespace tengen::engine
