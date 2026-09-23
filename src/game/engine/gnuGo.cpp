#include "engine/gnuGo.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace tengen::engine {

// GNU Go's strength is a level from 1 to 10, not a rank. Its default level 10 plays around 5k to 8k, and
// the lower levels play weaker by an amount nobody has measured. Until someone plays them against known
// ranks, the levels spread evenly from 20k to 6k, and any stronger skill gets level 10: GNU Go's best.
static constexpr Skill weakestLevel   = fromKyu(20); //!< Plays at level 1.
static constexpr Skill strongestLevel = fromKyu(6);  //!< Plays at level 10.
static constexpr int minLevel         = 1;
static constexpr int maxLevel         = 10;

//! Pick the level GNU Go plays a skill at.
//! \note This is the one place that knows how our skill scale maps onto GNU Go's levels.
static int level(const Skill skill) {
	const int weakest   = static_cast<int>(weakestLevel);
	const int strongest = static_cast<int>(strongestLevel);
	const int clamped   = std::clamp(static_cast<int>(skill), weakest, strongest);
	return minLevel + (clamped - weakest) * (maxLevel - minLevel) / (strongest - weakest);
}

GnuGo::GnuGo(std::string executable)
    : m_executable(std::move(executable)) {
}

void GnuGo::start(const unsigned boardSize, const tengen::Player botColour, const tengen::Skill botSkill) {
	// Our game forbids repeating a position with the same player to move: situational superko.
	// GNU Go only knows simple ko by default and would play moves our game refuses, which leaves the bot stuck.
	launch({.argv = {
	                m_executable,
	                "--mode",
	                "gtp",
	                "--level",
	                std::to_string(level(botSkill)),
	                "--situational-superko",
	        },
	        .requiredFiles = {m_executable},
	        .logFile       = "gnugo.log"},
	       boardSize, botColour);
}

} // namespace tengen::engine
