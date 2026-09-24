#pragma once

#include "engine/gnuGoConfig.hpp"
#include "engine/gtpEngine.hpp"

namespace tengen::engine {

//! Plays GNU Go. It needs nothing but its executable and runs on any machine.
class GnuGo : public GtpEngine {
public:
	explicit GnuGo(GnuGoConfig config);

	void start(unsigned boardSize, tengen::Player botColour) override;

private:
	GnuGoConfig m_config; //!< Where the engine relevant files physically lie and how the engine should play.
};

} // namespace tengen::engine
