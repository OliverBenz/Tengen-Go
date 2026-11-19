#include "network/core/tcpServer.hpp"

#include "tcpServerImpl.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace tengen {
namespace network::core {

TcpServer::TcpServer(Callbacks callbacks, std::uint16_t port)
    : m_pimpl(std::make_unique<Implementation>(std::move(callbacks), port)) {
}

TcpServer::~TcpServer() {
	stop();
}

bool TcpServer::start() {
	return m_pimpl->start();
}

void TcpServer::stop() {
	m_pimpl->stop();
}

bool TcpServer::send(ConnectionId connectionId, const Message& msg) {
	return m_pimpl->send(connectionId, msg);
}

void TcpServer::disconnect(ConnectionId connectionId) {
	m_pimpl->disconnect(connectionId);
}


} // namespace network::core
} // namespace tengen
