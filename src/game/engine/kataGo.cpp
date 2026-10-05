#include "engine/kataGo.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <string>
#include <utility>

namespace tengen::engine {

// Opening style of the imitated players:
// 'preaz' plays like humans did before AlphaZero changed how the opening is played
// 'rank'  like they do since.
static constexpr char profileStyle[] = "preaz_";

//! Name the humanSLProfile the engine should imitate, e.g. "preaz_5k".
//! \note This is the one place that knows how our skill scale maps onto KataGo's vocabulary.
static std::string humanProfile(const Skill rank) {
	return profileStyle + toString(rank);
}

static const char* koName(const Ko ko) {
	switch (ko) {
	case Ko::Simple:
		return "SIMPLE";
	case Ko::Situational:
		return "SITUATIONAL";
	case Ko::Positional:
		return "POSITIONAL";
	}

	assert(false);
	return "SIMPLE";
}

KataGo::KataGo(KataGoConfig config)
    : m_config(std::move(config)) {
}

void KataGo::start(const unsigned boardSize, const GameRules& rules, const tengen::Player botColour) {
	assert(acceptsKomi(rules.komi));

	const Skill rank = std::clamp(m_config.rank, KataGoConfig::weakestRank, KataGoConfig::strongestRank);

	const std::string executable = m_config.files.executable.string();
	const std::string model      = m_config.files.model.string();
	const std::string humanModel = m_config.files.humanModel.string();
	const std::string gtpConfig  = m_config.files.gtpConfig.string();

	// The rank overrides the profile the config file names, so one config serves every rank.
	launch({.argv = {
	                executable,
	                "gtp",
	                "-model",
	                model,
	                "-human-model",
	                humanModel,
	                "-config",
	                gtpConfig,
	                "-override-config",
	                "humanSLProfile=" + humanProfile(rank),
	        },
	        .requiredFiles = {executable, model, humanModel, gtpConfig},
	        .logFile       = "katago.log",
	        .setupCommands = {setRulesCommand(rules)}}, // Overrides the rules in the config file.
	       boardSize, rules, botColour);
}

bool KataGo::acceptsKomi(const float komi) {
	const float halfPoints = komi * 2.0f;
	return std::isfinite(komi) && halfPoints == std::round(halfPoints);
}

std::string KataGo::setRulesCommand(const GameRules& rules) {
	const bool area = rules.scoringMethod == Scoring::Area;

	// KataGo's suicide rule only covers several stones. A single stone is always illegal, like in our core.
	// No spaces, so the rules stay a single GTP argument.
	return std::format(R"(kata-set-rules {{"ko":"{}","scoring":"{}","tax":"{}","suicide":{},"friendlyPassOk":{}}})",
	                   koName(rules.koRule),
	                   area ? "AREA" : "TERRITORY",
	                   area ? "NONE" : "SEKI",
	                   rules.suicideLegal ? "true" : "false",
	                   area ? "true" : "false");
}

} // namespace tengen::engine
