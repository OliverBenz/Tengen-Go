#include "tengen/eventHub.hpp"
#include "tengen/IAppSignal.hpp"

#include <algorithm>

namespace tengen::app {

EventHub::EventHub(const IDispatcher& dispatcher) : m_dispatcher{dispatcher} {
}

void EventHub::subscribe(IAppSignalListener* listener, uint64_t signalMask) {
	std::lock_guard<std::mutex> lock(m_listenerMutex);

	m_signalListeners.push_back({listener, signalMask});
}

void EventHub::unsubscribe(IAppSignalListener* listener) {
	std::lock_guard<std::mutex> lock(m_listenerMutex);

	m_signalListeners.erase(
	        std::remove_if(m_signalListeners.begin(), m_signalListeners.end(), [&](const SignalListenerEntry& e) { return e.listener == listener; }),
	        m_signalListeners.end());
}

void EventHub::signal(const AppSignal signal) {
	m_pendingUpdates.fetch_or(signal);

	if (!m_drainScheduled.exchange(true)) {
		m_dispatcher.post([this] { drain(); });
	}
}

void EventHub::drain() {
	// Get and reset pending updates
	const AppSignalMask pending = m_pendingUpdates.exchange(0u);
	if (pending == 0) {
		m_drainScheduled.store(false);
		return;
	}

	// Get all current listeners
	std::vector<SignalListenerEntry> listeners;
	{
		std::lock_guard lock(m_listenerMutex);
		listeners = m_signalListeners;
	}

	// Drain events to listeners
	for (const auto& [listener, signalMask]: listeners) {
		const auto relevantEvents = signalMask & pending;
		if (relevantEvents != 0) {
			listener->onAppEvent(relevantEvents);
		}
	}

	// Drain finished
	m_drainScheduled.store(false);

	// Check anything happenened since
	if (m_pendingUpdates.load() != 0 && !m_drainScheduled.exchange(true)) {
		m_dispatcher.post([this] { drain(); });
	}
}

} // namespace tengen::app
