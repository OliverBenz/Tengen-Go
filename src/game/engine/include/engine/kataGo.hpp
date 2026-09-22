#pragma once

#include "engine/IEngineListener.hpp"
#include "engine/botMove.hpp"
#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <memory>
#include <string>

namespace tengen::engine {

//! Where the engine and its assets live. This is deployment configuration and says nothing about how strong the bot plays: the strength comes in per game as a Skill.
struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
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

	//! Bring katago up and set the game up. Returns at once; answers with onEngineReady().
	//! \note The engine only imitates ranks it was trained on, so a skill outside that range plays at the closest one it has.
	void start(const LaunchConfig& config, unsigned boardSize, tengen::Player botColour, tengen::Skill botSkill);

	void stop();            //!< No listener callbacks once this returns. Never call from a callback.
	bool isRunning() const; //!< False before start() and from the moment stop() begins.

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
