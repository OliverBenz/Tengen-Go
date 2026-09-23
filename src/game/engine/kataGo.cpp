#include "engine/kataGo.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace tengen::engine {

// The human model imitates ranks from 20k to 9d and nothing outside of it. A weaker bot than its
// floor is a matter of handicap stones rather than of profile, so a skill below it plays at 20k.
static constexpr Skill weakestProfile   = fromKyu(20);
static constexpr Skill strongestProfile = fromDan(9);

// Opening style of the imitated players:
// 'preaz' plays like humans did before AlphaZero changed how the opening is played
// 'rank'  like they do since.
static constexpr char profileStyle[] = "preaz_";

//! Name the humanSLProfile the engine should imitate, e.g. "preaz_5k".
//! \note This is the one place that knows how our skill scale maps onto KataGo's vocabulary.
static std::string humanProfile(const Skill skill) {
	return profileStyle + toString(std::clamp(skill, weakestProfile, strongestProfile));
}

KataGo::KataGo(LaunchConfig config)
    : m_config(std::move(config)) {
}

void KataGo::start(const unsigned boardSize, const tengen::Player botColour, const tengen::Skill botSkill) {
	// The strength overrides the profile the config file names, so one config serves every rank.
	launch({.argv          = {m_config.executable,
	                          "gtp",
	                          "-model", m_config.model,
	                          "-human-model", m_config.modelHuman,
	                          "-config", m_config.config,
	                          "-override-config", "humanSLProfile=" + humanProfile(botSkill)},
	        .requiredFiles = {m_config.executable, m_config.model, m_config.modelHuman, m_config.config},
	        .logFile       = "katago.log"},
	       boardSize, botColour);
}

} // namespace tengen::engine
