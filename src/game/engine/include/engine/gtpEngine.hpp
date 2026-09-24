#pragma once

#include "engine/IEngineListener.hpp"
#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tengen::engine {

//! Drives a bot engine process over GTP.
//! Engines only differ in how they are brought up: a derived engine implements start() and hands its launch to launch().
//! \note    Long requests run on the engine's own thread and answer through the listener.
//! \example Usage: registerListener(), then start() once. Call stop() to shut down.
class GtpEngine {
public:
	virtual ~GtpEngine();

	GtpEngine(const GtpEngine&)            = delete;
	GtpEngine& operator=(const GtpEngine&) = delete;
	GtpEngine(GtpEngine&&)                 = delete;
	GtpEngine& operator=(GtpEngine&&)      = delete;

	//! Register a single listener. Returns false if already registered.
	bool registerListener(IEngineListener* listener);

	//! Bring the engine up and set the game up. Answers with EngineListener.
	virtual void start(unsigned boardSize, tengen::Player botColour) = 0;

	//! No listener callbacks once this returns.
	//! \note Never call from a callback.
	void stop();
	bool isRunning() const;

	// Answer immediately on the calling thread. Only send while the engine is not thinking.
	bool place(tengen::Coord pos);
	bool pass();
	bool resign();

	void genmove(); //!< Returns at once; answers with onMoveGenerated().

protected:
	GtpEngine();

	//! How to bring one particular engine up.
	struct Launch {
		std::vector<std::string> argv;          //!< The executable, then its arguments.
		std::vector<std::string> requiredFiles; //!< The engine is not even started while one of these is missing.
		std::string logFile;                    //!< Takes over the engine's stderr.
	};

	void launch(Launch command, unsigned boardSize, tengen::Player botColour); //!< What every start() comes down to.

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide the engine process and the thread it runs on.
};

} // namespace tengen::engine
