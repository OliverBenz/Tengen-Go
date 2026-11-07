#include "tengen/sessionManager.hpp"

#include "botSession.hpp"
#include "networkSession.hpp"
#include "openSession.hpp"

#include <cassert>

namespace tengen::app {

SessionManager::SessionManager()  = default;
SessionManager::~SessionManager() = default;

void SessionManager::startOpen(const GameConfig& config) {
	shutdown();
	m_session = std::make_unique<Session>(std::in_place_type<OpenSession>, config);
}

void SessionManager::startBot(const GameConfig& config, const engine::EngineConfig& engineConfig, const bool humanPlaysBlack) {
	shutdown();
	m_session = std::make_unique<Session>(std::in_place_type<BotSession>, config, engine::makeEngine(engineConfig), humanPlaysBlack);
}

void SessionManager::startNetwork() {
	shutdown();
	m_session = std::make_unique<Session>(std::in_place_type<NetworkSession>);
}

void SessionManager::connect(const std::string& hostIp) {
	auto* session = std::get_if<NetworkSession>(m_session.get());
	assert(session); // Needs startNetwork() first.
	session->connect(hostIp);
}

bool SessionManager::host(const GameConfig& config, const Player hostColour) {
	auto* session = std::get_if<NetworkSession>(m_session.get());
	assert(session); // Needs startNetwork() first.
	return session->host(config, hostColour);
}

void SessionManager::shutdown() {
	if (m_session) {
		std::visit([](auto& active) { active.shutdown(); }, *m_session);
		m_session.reset();
	}
}

IGameSession* SessionManager::game() {
	if (!m_session) {
		return nullptr;
	}
	return std::visit([](auto& active) -> IGameSession* { return &active; }, *m_session);
}

IChatSession* SessionManager::chat() {
	return std::get_if<NetworkSession>(m_session.get()); // Only network games have a chat.
}

INetworkSession* SessionManager::network() {
	return std::get_if<NetworkSession>(m_session.get());
}

} // namespace tengen::app
