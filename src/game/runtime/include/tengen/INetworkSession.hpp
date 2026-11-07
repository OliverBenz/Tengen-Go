#pragma once

#include "model/player.hpp"
#include "network/client.hpp"
#include "tengen/IAppSignal.hpp"

#include <cstdint>
#include <optional>

namespace tengen::app {

enum class ConnectionPhase : std::uint8_t {
	Idle,       //!< No connection. The end reason tells how the last one ended.
	Connecting, //!< Waiting for the welcome.
	Connected,  //!< The server welcomed us.
};

//! The connection as a session knows it: what the server told us about ourselves, and how the connection ended.
struct ConnectionState {
	ConnectionPhase phase;
	std::optional<Player> localPlayer;                  //!< Our colour while connected. Nullopt: not welcomed, or we only watch.
	std::optional<network::DisconnectReason> endReason; //!< How the last connection ended. Nullopt while one runs.
};

class INetworkSession : public IAppSignalSource {
public:
	virtual ~INetworkSession() = default;

	virtual ConnectionState connection() const = 0; //!< A copy, to read off the lock.
};

} // namespace tengen::app
