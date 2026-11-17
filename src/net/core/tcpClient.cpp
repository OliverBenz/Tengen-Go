#include "network/core/tcpClient.hpp"

#include "tcpClientImpl.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tengen {
namespace network::core {

TcpClient::TcpClient(Callbacks callbacks)
    : m_pimpl(std::make_unique<Implementation>(std::move(callbacks))) {
}

TcpClient::~TcpClient() = default;

bool TcpClient::connect(std::string host, std::uint16_t port, std::optional<Message> firstMessage) {
	return m_pimpl->connect(std::move(host), port, std::move(firstMessage));
}

bool TcpClient::disconnect() {
	return m_pimpl->disconnect();
}

bool TcpClient::isConnected() const {
	return m_pimpl->isConnected();
}

bool TcpClient::send(const Message& message) {
	return m_pimpl->send(message);
}

} // namespace network::core
} // namespace tengen
