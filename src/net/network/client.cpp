#include "network/client.hpp"

#include "clientImpl.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace tengen::network {

Client::Client()
    : m_pimpl(std::make_unique<Implementation>()) {
}

Client::~Client() = default;

bool Client::registerHandler(IClientHandler* handler) {
	return m_pimpl->registerHandler(handler);
}

bool Client::connect(const std::string& host) {
	return m_pimpl->connect(host, core::DEFAULT_PORT);
}

bool Client::connect(const std::string& host, std::uint16_t port) {
	return m_pimpl->connect(host, port);
}

void Client::disconnect() {
	m_pimpl->disconnect();
}

bool Client::isConnected() const {
	return m_pimpl->isConnected();
}

bool Client::send(const ClientEvent& event) {
	return m_pimpl->send(event);
}

SessionId Client::sessionId() const {
	return m_pimpl->sessionId();
}

} // namespace tengen::network
