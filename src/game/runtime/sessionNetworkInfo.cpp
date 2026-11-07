#include "sessionNetworkInfo.hpp"

namespace tengen::app {

void SessionNetworkInfo::setConnecting() {
	m_state.phase = ConnectionPhase::Connecting;
	m_state.localPlayer.reset();
	m_state.endReason.reset();
}

void SessionNetworkInfo::setConnected(const std::optional<Player> player) {
	m_state.phase       = ConnectionPhase::Connected;
	m_state.localPlayer = player;
}

void SessionNetworkInfo::setIdle(const network::DisconnectReason reason) {
	m_state.phase = ConnectionPhase::Idle;
	m_state.localPlayer.reset();
	m_state.endReason = reason;
}

const ConnectionState& SessionNetworkInfo::state() const {
	return m_state;
}

} // namespace tengen::app
