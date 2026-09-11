#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include "subProcess.hpp"

#include <string>

struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
};


// TODO: I could generalize one more step.
// KataGoEngine -> GtpEngine
// Only Launchconfig is specific? player commands are GTP specified.

class KataGo {
public:
	KataGo()                         = default;
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

private:
	SubProcess m_process;                              //!< The engine process.
	tengen::Player m_botColour{tengen::Player::Black}; //!< Colour the bot plays. The player takes the other one.
	unsigned m_boardSize{9u};                          //!< Board size the game was started with.
};
