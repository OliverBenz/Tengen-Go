#include "engine/kataGo.hpp"

#include <algorithm>
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

KataGo::KataGo(KataGoConfig config)
    : m_config(std::move(config)) {
}

void KataGo::start(const unsigned boardSize, const tengen::Player botColour) {
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
	        .logFile       = "katago.log"},
	       boardSize, botColour);
}

} // namespace tengen::engine
