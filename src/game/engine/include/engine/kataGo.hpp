#pragma once

#include "engine/gtpEngine.hpp"
#include "engine/kataGoConfig.hpp"

#include <string>

namespace tengen::engine {

//! Plays KataGo, imitating a human of the configured rank.
class KataGo : public GtpEngine {
public:
	explicit KataGo(KataGoConfig config);

	void start(unsigned boardSize, const GameRules& rules, tengen::Player botColour) override;

	//! KataGo's kata-set-rules command for the ko, scoring and suicide rules. Komi goes over GTP.
	static std::string setRulesCommand(const GameRules& rules);

private:
	KataGoConfig m_config; //!< Where the engine relevant files physically lie and how the engine should play.
};

} // namespace tengen::engine
