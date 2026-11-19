#pragma once

#include "engine/gtpEngine.hpp"

#include "subProcess.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tengen::engine {

class GtpEngine::Implementation {
public:
	Implementation() = default;

	bool registerListener(IEngineListener* listener);

	void start(Launch command, const GameConfig& gameConfig, tengen::Player botColour);
	void stop();
	bool isRunning() const;

	bool place(tengen::Coord pos);
	bool pass();

	void genmove();

private:
	void post(std::function<void()> request); //!< Hand one request to the engine thread. Dropped once stopped.
	void engineLoop();                        //!< Engine thread: run the posted requests until stop().
	bool canNotify() const;                   //!< False once stop() silenced the answers.

private:
	bool launch(const Launch& command);
	bool setupGame(const std::vector<std::string>& engineCommands);
	bool sendCommand(const std::string& command, std::string& response); //!< Send one GTP command and wait for its response.

private:
	SubProcess m_process;
	IEngineListener* m_listener{nullptr};
	tengen::Player m_botColour{tengen::Player::Black}; //!< The player takes the other one.
	GameConfig m_gameConfig{.boardSize = 9u, .rules = fromRuleSet(RuleSet::Japanese)};

	std::atomic<bool> m_running{false}; //!< Engine thread running.
	std::thread m_engineThread;
	std::mutex m_requestMutex;
	std::condition_variable m_requestReady;
	std::function<void()> m_pendingRequest; //!< Request waiting to be picked up. At most one.
};

} // namespace tengen::engine
