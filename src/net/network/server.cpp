#include "network/server.hpp"

#include "serverImpl.hpp"

#include <memory>

namespace tengen::network {

Server::Server()
    : m_pimpl(std::make_unique<Implementation>(core::DEFAULT_PORT)) {
}

Server::Server(std::uint16_t port)
    : m_pimpl(std::make_unique<Implementation>(port)) {
}

Server::~Server() {
	stop();
}

bool Server::start() {
	return m_pimpl->start();
}

void Server::stop() {
	m_pimpl->stop();
}

bool Server::registerHandler(IServerHandler* handler) {
	return m_pimpl->registerHandler(handler);
}

void Server::setFirstSeat(const Seat seat) {
	m_pimpl->setFirstSeat(seat);
}

bool Server::send(SessionId sessionId, const ServerEvent& event) {
	return m_pimpl->send(sessionId, event);
}

bool Server::broadcast(const ServerEvent& event) {
	return m_pimpl->broadcast(event);
}

} // namespace tengen::network
