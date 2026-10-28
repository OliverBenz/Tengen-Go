#include "sessionManager.hpp"

#include <cassert>
#include <type_traits>

namespace tengen::network {

SessionId SessionManager::add(core::ConnectionId connectionId) {
	const auto existing = m_connectionToSession.find(connectionId);
	if (existing != m_connectionToSession.end()) {
		const auto it = m_sessions.find(existing->second);
		if (it != m_sessions.end()) {
			it->second.isActive = true;
			return it->second.sessionId;
		}
		m_connectionToSession.erase(existing);
	}

	const auto sessionId = generateSessionId();
	SessionContext context{
	        .connectionId = connectionId,
	        .sessionId    = sessionId,
	        .seat         = Seat::None,
	        .isActive     = true,
	};
	m_sessions.emplace(sessionId, context);
	m_connectionToSession.emplace(connectionId, sessionId);
	return sessionId;
}

void SessionManager::remove(SessionId sessionId) {
	const auto it = m_sessions.find(sessionId);
	if (it == m_sessions.end()) {
		return;
	}
	m_connectionToSession.erase(it->second.connectionId);
	m_sessions.erase(it);
}

SessionId SessionManager::getSessionId(core::ConnectionId connectionId) const {
	const auto it = m_connectionToSession.find(connectionId);
	if (it == m_connectionToSession.end()) {
		return SessionId::Empty;
	}
	return it->second;
}

core::ConnectionId SessionManager::getConnectionId(SessionId sessionId) const {
	const auto it = m_sessions.find(sessionId);
	if (it == m_sessions.end()) {
		return core::ConnectionId::Empty;
	}
	return it->second.connectionId;
}

core::ConnectionId SessionManager::getConnectionIdBySeat(Seat seat) const {
	assert(seat == Seat::Black || seat == Seat::White);

	for (const auto& [_, context]: m_sessions) {
		if (context.isActive && context.seat == seat) {
			return context.connectionId;
		}
	}
	return core::ConnectionId::Empty;
}

Seat SessionManager::getSeat(SessionId sessionId) const {
	const auto it = m_sessions.find(sessionId);
	if (it == m_sessions.end()) {
		return Seat::None;
	}
	return it->second.seat;
}

void SessionManager::setSeat(SessionId sessionId, Seat seat) {
	const auto it = m_sessions.find(sessionId);
	if (it == m_sessions.end()) {
		return;
	}
	it->second.seat = seat;
}

void SessionManager::setDisconnected(SessionId sessionId) {
	const auto it = m_sessions.find(sessionId);
	if (it == m_sessions.end()) {
		return;
	}
	it->second.isActive = false;
}

void SessionManager::forEachSession(const std::function<void(const SessionContext&)>& visitor) const {
	for (const auto& [_, context]: m_sessions) {
		visitor(context);
	}
}

SessionId SessionManager::generateSessionId() const {
	static std::underlying_type_t<SessionId> nextSessionId{1};

	auto candidate = nextSessionId;
	if (SessionId{candidate} == SessionId::Empty) {
		candidate = 1;
	}
	while (m_sessions.contains(SessionId{candidate})) {
		++candidate;
		if (SessionId{candidate} == SessionId::Empty) {
			++candidate;
		}
	}

	nextSessionId = candidate + 1;
	if (SessionId{nextSessionId} == SessionId::Empty) {
		nextSessionId = 1;
	}
	return SessionId{candidate};
}


} // namespace tengen::network
