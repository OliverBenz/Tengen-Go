#pragma once

#include "core/SafeQueue.hpp"
#include "core/eventHub.hpp"
#include "core/gameEvent.hpp"
#include "core/gameState.hpp"
#include "model/gameRules.hpp"

namespace tengen {

using EventQueue = SafeQueue<GameEvent>;

//! Core game setup.
//! This owns the rules loop and emits deltas; external code should only push events and listen.
class Game {
public:
	//! Setup a game of certain board size without starting the game loop.
	Game(std::size_t boardSize, const GameRules& rules);

	void run();                      //!< Handle events until a ShutdownEvent (blocking). Keeps running after the game ended.
	void pushEvent(GameEvent event); //!< Push an event to the event queue.

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
