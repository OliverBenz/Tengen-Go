#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include "botMove.hpp"

#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace tengen::engine {

struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
};

class SubProcess;

class KataGo {
public:
	KataGo();
	KataGo(const KataGo&)            = delete;
	KataGo& operator=(const KataGo&) = delete;
	~KataGo();

	bool start(const LaunchConfig& config); //!< Start katago with the given configuration.
	void stop();                            //!< Stop the subprocess.

	// Note: Validate the move is legal beforehand.
	bool startGame(unsigned boardSize, tengen::Player botColour);
	bool place(tengen::Coord pos);
	bool pass();
	bool resign();

	//! Let the engine pick its move on its own thread. Calls back with the result once it answers.
	void genmove(std::function<void(bool ok, BotMove move)> callback);

private:
	bool sendCommand(const std::string& command, std::string& response); //!< Send one GTP command and wait for its response.

private:
	std::unique_ptr<SubProcess> m_process{nullptr};    //!< The engine process.
	tengen::Player m_botColour{tengen::Player::Black}; //!< Colour the bot plays. The player takes the other one.
	unsigned m_boardSize{9u};                          //!< Board size the game was started with.
	std::thread m_genmoveThread;                       //!< Runs the in flight genmove request. Retired by stop().
};

} // namespace tengen::engine
