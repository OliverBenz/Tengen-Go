#pragma once

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include "botMove.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace tengen::engine {

struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
};

//! Callback interface invoked on the engine's worker thread.
//! \note Keep handlers lightweight.
class IEngineListener {
public:
	virtual ~IEngineListener()                        = default;
	virtual void onEngineReady()                      = 0; //!< The engine is up and the game is set up.
	virtual void onMoveGenerated(const BotMove& move) = 0; //!< The engine picked its move and played it on its own board.
	virtual void onEngineFailed()                     = 0; //!< The engine cannot answer anymore.
};

class SubProcess;

//! Drives a KataGo process over GTP.
//! Long requests run on a worker thread of this class and answer through the listener, so callers never
//! block on a search. Only one request is ever in flight.
class KataGo {
public:
	KataGo();
	KataGo(const KataGo&)            = delete;
	KataGo& operator=(const KataGo&) = delete;
	~KataGo();

	bool registerListener(IEngineListener* listener); //!< Register a single listener. Returns false if one is already registered.

	//! Bring katago up and set the game up. Returns at once; the listener hears onEngineReady() once it is up.
	void start(const LaunchConfig& config, unsigned boardSize, tengen::Player botColour);
	void stop(); //!< Stop the subprocess and retire the worker. No listener callbacks after this.

	// Short round trips that run no search, so they answer on the calling thread.
	// Note: Validate the move is legal beforehand, and only send while the engine is not thinking.
	bool place(tengen::Coord pos);
	bool pass();
	bool resign();

	//! Let the engine pick its move on the worker thread. The listener hears onMoveGenerated() once it answers.
	void genmove();

private:
	bool launch(const LaunchConfig& config);                             //!< Start the process and make sure it speaks GTP.
	bool setupGame();                                                    //!< Set the engine's board up for the game we start.
	bool sendCommand(const std::string& command, std::string& response); //!< Send one GTP command and wait for its response.

	void post(std::function<void()> request); //!< Hand one request to the worker. Only one is ever in flight.
	void workerLoop();                        //!< Runs the posted requests until stop().
	bool canNotify() const;                   //!< False once stop() killed the request the answer belongs to.

private:
	std::unique_ptr<SubProcess> m_process{nullptr};    //!< The engine process.
	IEngineListener* m_listener{nullptr};              //!< Hears the answers of the requests run on the worker.
	tengen::Player m_botColour{tengen::Player::Black}; //!< Colour the bot plays. The player takes the other one.
	unsigned m_boardSize{9u};                          //!< Board size the game was started with.

	std::thread m_worker;                   //!< Runs the requests. Started by start(), retired by stop().
	std::atomic<bool> m_running{false};     //!< Worker running. Cleared by stop(), which also silences the answers.
	std::mutex m_requestMutex;              //!< Guards the pending request.
	std::condition_variable m_requestReady; //!< Wakes the worker on a new request or on stop().
	std::function<void()> m_pendingRequest; //!< Request waiting to be picked up. At most one.
};

} // namespace tengen::engine
