#pragma once

#include "IAppSignal.hpp"

#include <atomic>
#include <mutex>
#include <vector>

namespace tengen::app {

//! Allows external components to be updated on internal game events.
class EventHub {
public:
	//! Takes a dispatcher used to asynchronously the update functions on the target thread.
	EventHub(const IDispatcher& dispatcher);

	void subscribe(IAppSignalListener* listener, uint64_t signalMask);
	void unsubscribe(IAppSignalListener* listener);
	void signal(AppSignal signal); //!< Signal a game event.

private:
	void drain();

private:
	struct SignalListenerEntry {
		IAppSignalListener* listener; //!< Pointer to the listener.
		uint64_t signalMask;          //!< What events the listener cares about.
	};
	std::mutex m_listenerMutex;
	std::vector<SignalListenerEntry> m_signalListeners;

	const IDispatcher& m_dispatcher;                 //!< Helper class used to asynchronusly call the event-update function on the target thread.
	std::atomic<AppSignalMask> m_pendingUpdates{0u}; //!< Bitmask of app signals that have been sent.
	std::atomic<bool> m_drainScheduled{false}; //!< On app events, we schedule that the Listeners should drain signals (apply the updates given a signalMask).
};

} // namespace tengen::app
