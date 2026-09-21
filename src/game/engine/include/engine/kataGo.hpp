#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include "botMove.hpp"

#include <memory>
#include <string>

namespace tengen::engine {

struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
};

//! Callback interface invoked on the engine's request thread.
//! \note Keep handlers lightweight.
class IEngineListener {
public:
	virtual ~IEngineListener() = default;

	virtual void onEngineReady()                      = 0;
	virtual void onMoveGenerated(const BotMove& move) = 0; //!< The engine already played the move on its own board.
	virtual void onEngineFailed()                     = 0;
};

//! Drives a KataGo process over GTP.
//! \note    Long requests run on the engine's own thread and answer through the listener.
//! \example Usage: registerListener(), then start() once. Call stop() to shut down.
class KataGo {
public:
	KataGo();
	~KataGo();

	KataGo(const KataGo&)            = delete;
	KataGo& operator=(const KataGo&) = delete;
	KataGo(KataGo&&)                 = delete;
	KataGo& operator=(KataGo&&)      = delete;

	bool registerListener(IEngineListener* listener); //!< Register a single listener. Returns false if already registered.

	void start(const LaunchConfig& config, unsigned boardSize, tengen::Player botColour); //!< Bring katago up and set the game up. Returns at once; answers with onEngineReady().
	void stop();                                                                          //!< No listener callbacks once this returns. Never call from a callback.
	bool isRunning() const;                                                               //!< False before start() and from the moment stop() begins.

	// Short round trips that answer on the calling thread. Only send while the engine is not thinking.
	bool place(tengen::Coord pos);
	bool pass();
	bool resign();

	void genmove(); //!< Returns at once; answers with onMoveGenerated().

private:
	class Implementation;
	std::unique_ptr<Implementation> m_pimpl; //!< Pimpl to hide the engine process and the thread it runs on.
};

} // namespace tengen::engine
