#pragma once

#include "core/SafeQueue.hpp"
#include "core/eventHub.hpp"
#include "core/gameEvent.hpp"
#include "core/gameState.hpp"
#include "model/gameRules.hpp"

namespace tengen {

using EventQueue = SafeQueue<GameEvent>;

//! Core game setup.
//! You first register as a game (signal/state) listener to get notified on game changes.
//! Then, you push events. The game will forward these to the GameState class and signal you on updates.
class Game {
public:
	//! Setup a game of certain board size without starting the game loop.
	Game(std::size_t boardSize, const GameRules& rules);

	void run();                      //!< Handle events until a ShutdownEvent (blocking). Keeps running after the game ended.
	void pushEvent(GameEvent event); //!< Push an event to the event queue.

	// TODO: Remove. Callers know the size from construction, and reading it off the game thread races with run().
	// TODO: We should signal on game start to make the event stream complete. Let the listeners know game start+rules+boardSize, etc.
	std::size_t boardSize() const;

public:
	void subscribeSignals(IGameSignalListener* listener, uint64_t signalMask);
	void unsubscribeSignals(IGameSignalListener* listener);
	void subscribeState(IGameStateListener* listener);
	void unsubscribeState(IGameStateListener* listener);

private:
	void handleEvent(const PutStoneEvent& event);
	void handleEvent(const PassEvent& event);
	void handleEvent(const ResignEvent& event);
	void handleEvent(const ShutdownEvent& event);

private:
	bool m_running{false}; //!< Event loop runs until a ShutdownEvent.

	GameState m_state;       //!< Position and rules. All position changes go through here.
	EventQueue m_eventQueue; //!< Queue of internal game events we have to handle.
	EventHub m_eventHub;     //!< Hub to signal updates of the game state to external components.
};

} // namespace tengen
