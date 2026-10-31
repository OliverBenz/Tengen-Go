#include "tengen/eventHub.hpp"

#include <algorithm>

namespace tengen::app {

void EventHub::subscribe(IAppSignalListener* listener, AppSignalMask signalMask) {
	std::lock_guard<std::mutex> lock(m_listenerMutex);

	m_signalListeners.push_back({listener, signalMask});
}

void EventHub::unsubscribe(IAppSignalListener* listener) {
	std::lock_guard<std::mutex> lock(m_listenerMutex);

	m_signalListeners.erase(
	        std::remove_if(m_signalListeners.begin(), m_signalListeners.end(), [&](const SignalListenerEntry& e) { return e.listener == listener; }),
	        m_signalListeners.end());
}

void EventHub::signal(AppSignalMask mask) {
	std::lock_guard<std::mutex> lock(m_listenerMutex);

	for (const auto& [listener, signalMask]: m_signalListeners) {
		if (const auto subscribed = signalMask & mask) {
			listener->onAppEvent(subscribed);
		}
	}
}

} // namespace tengen::app
