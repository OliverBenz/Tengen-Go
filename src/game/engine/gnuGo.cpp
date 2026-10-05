#include "engine/gnuGo.hpp"

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>

namespace tengen::engine {

static const char* koArgument(const Ko ko) {
	switch (ko) {
	case Ko::Simple:
		return "--simple-ko";
	case Ko::Situational:
		return "--situational-superko";
	case Ko::Positional:
		return "--positional-superko";
	}

	assert(false);
	return "--simple-ko";
}

GnuGo::GnuGo(GnuGoConfig config)
    : m_config(std::move(config)) {
}

void GnuGo::start(const unsigned boardSize, const GameRules& rules, const tengen::Player botColour) {
	const int level              = std::clamp(m_config.level, GnuGoConfig::weakestLevel, GnuGoConfig::strongestLevel);
	const std::string executable = m_config.files.executable.string();

	std::vector<std::string> argv{executable, "--mode", "gtp", "--level", std::to_string(level)};

	// GNU Go takes the rules on its command line
	const std::vector<std::string> rulesArgv = ruleArguments(rules);
	argv.insert(argv.end(), rulesArgv.begin(), rulesArgv.end());

	launch({.argv = std::move(argv), .requiredFiles = {executable}, .logFile = "gnugo.log", .setupCommands = {}}, boardSize, rules, botColour);
}

std::vector<std::string> GnuGo::ruleArguments(const GameRules& rules) {
	return {
	        rules.scoringMethod == Scoring::Area ? "--chinese-rules" : "--japanese-rules",
	        koArgument(rules.koRule),
	        // Like our core, --allow-suicide keeps single-stone suicide illegal. --allow-all-suicide would not.
	        rules.suicideLegal ? "--allow-suicide" : "--forbid-suicide",
	};
}

} // namespace tengen::engine
