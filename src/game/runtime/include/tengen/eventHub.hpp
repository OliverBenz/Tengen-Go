#pragma once

#include "IAppSignal.hpp"

#include <mutex>
#include <vector>

namespace tengen::app {

//! Allows external components to be updated on internal game events.
class EventHub {
	struct SignalListenerEntry {
		IAppSignalListener* listener; //!< Pointer to the listener.
		AppSignalMask signalMask;     //!< What events the listener cares about.
	};

public:
	void subscribe(IAppSignalListener* listener, AppSignalMask signalMask);
	void unsubscribe(IAppSignalListener* listener);
	void signal(AppSignalMask mask); //!< Signal one or more game events in a single call.

private:
	std::mutex m_listenerMutex;
	std::vector<SignalListenerEntry> m_signalListeners;
};

} // namespace tengen::app
