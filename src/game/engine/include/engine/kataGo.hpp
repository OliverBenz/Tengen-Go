#pragma once

#include "engine/gtpEngine.hpp"
#include "engine/kataGoConfig.hpp"

namespace tengen::engine {

//! Plays KataGo, imitating a human of the configured rank.
class KataGo : public GtpEngine {
public:
	explicit KataGo(KataGoConfig config);

	void start(unsigned boardSize, tengen::Player botColour) override;

private:
	KataGoConfig m_config; //!< Where the engine relevant files physically lie and how the engine should play.
};

} // namespace tengen::engine
