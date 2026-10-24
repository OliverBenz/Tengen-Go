#include "tengen/sessionNetworkInfo.hpp"

namespace tengen::app {

void SessionNetworkInfo::connecting() {
	m_phase = ConnectionPhase::Connecting;
	m_localPlayer.reset();
	m_endReason.reset();
}

void SessionNetworkInfo::update(const std::optional<Player> player) {
	m_phase       = ConnectionPhase::Connected;
	m_localPlayer = player;
}

void SessionNetworkInfo::update(const network::DisconnectReason reason) {
	m_phase = ConnectionPhase::Idle;
	m_localPlayer.reset();
	m_endReason = reason;
}

ConnectionPhase SessionNetworkInfo::getPhase() const {
	return m_phase;
}

std::optional<Player> SessionNetworkInfo::getLocalPlayer() const {
	return m_localPlayer;
}

std::optional<network::DisconnectReason> SessionNetworkInfo::getEndReason() const {
	return m_endReason;
}

} // namespace tengen::app
