#pragma once

#include "engine/gnuGoConfig.hpp"
#include "engine/gtpEngine.hpp"

#include <string>
#include <vector>

namespace tengen::engine {

//! Plays GNU Go. It needs nothing but its executable and runs on any machine.
class GnuGo : public GtpEngine {
public:
	explicit GnuGo(GnuGoConfig config);

	void start(unsigned boardSize, const GameRules& rules, tengen::Player botColour) override;

	//! GNU Go's command-line options for the ko, scoring and suicide rules. Komi goes over GTP.
	static std::vector<std::string> ruleArguments(const GameRules& rules);

private:
	GnuGoConfig m_config; //!< Where the engine relevant files physically lie and how the engine should play.
};

} // namespace tengen::engine
