#pragma once

#include "engine/gtpEngine.hpp"

#include <string>

namespace tengen::engine {

//! Plays GNU Go. It needs nothing but its executable and runs on any machine.
//! \note GNU Go's strength is a level rather than a rank, so the skill only picks roughly how strong it plays.
class GnuGo : public GtpEngine {
public:
	explicit GnuGo(std::string executable);

	void start(unsigned boardSize, tengen::Player botColour, tengen::Skill botSkill) override;

private:
	std::string m_executable;
};

} // namespace tengen::engine
