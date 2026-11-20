#pragma once

#include "model/player.hpp"
#include "network/client.hpp"
#include "tengen/INetworkSession.hpp"

#include <optional>

namespace tengen::app {

//! Moves the connection state along. Only these transitions change it.
class SessionNetworkInfo {
public:
	void setConnecting();                            //!< A new attempt. Forgets the last one.
	void setConnected(std::optional<Player> player); //!< Welcomed as this player. Nullopt: we only watch.
	void setIdle(network::DisconnectReason reason);  //!< Ended. The reason stays until the next attempt.

	const ConnectionState& state() const;

private:
	ConnectionState m_state{.phase = ConnectionPhase::Idle, .localPlayer = std::nullopt, .endReason = std::nullopt};
};

} // namespace tengen::app
