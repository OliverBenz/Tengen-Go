#include "engine/gnuGo.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace tengen::engine {

GnuGo::GnuGo(GnuGoConfig config)
    : m_config(std::move(config)) {
}

void GnuGo::start(const unsigned boardSize, const tengen::Player botColour) {
	const int level              = std::clamp(m_config.level, GnuGoConfig::weakestLevel, GnuGoConfig::strongestLevel);
	const std::string executable = m_config.files.executable.string();

	// Our game forbids repeating a position with the same player to move: situational superko.
	// GNU Go only knows simple ko by default and would play moves our game refuses, which leaves the bot stuck.
	launch({.argv = {
	                executable,
	                "--mode",
	                "gtp",
	                "--level",
	                std::to_string(level),
	                "--situational-superko",
	        },
	        .requiredFiles = {executable},
	        .logFile       = "gnugo.log"},
	       boardSize, botColour);
}

} // namespace tengen::engine
