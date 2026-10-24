#pragma once

#include "model/player.hpp"
#include "network/client.hpp"

#include <cstdint>
#include <optional>

namespace tengen::app {

enum class ConnectionPhase : std::uint8_t {
	Idle,       //!< No connection. getEndReason() tells how the last one ended.
	Connecting, //!< Waiting for the welcome.
	Connected,  //!< The server welcomed us.
};

//! The connection as a session knows it: what the server told us about ourselves, and how the connection ended.
class SessionNetworkInfo {
public:
	void connecting();                             //!< A new attempt. Forgets the last one.
	void update(std::optional<Player> player);     //!< Welcomed as this player. Nullopt: we only watch.
	void update(network::DisconnectReason reason); //!< Ended. The reason stays until the next attempt.

	ConnectionPhase getPhase() const;
	std::optional<Player> getLocalPlayer() const;                  //!< Our colour while connected. Nullopt: not welcomed, or we only watch.
	std::optional<network::DisconnectReason> getEndReason() const; //!< How the last connection ended. Nullopt while one runs.

private:
	ConnectionPhase m_phase{ConnectionPhase::Idle};
	std::optional<Player> m_localPlayer;
	std::optional<network::DisconnectReason> m_endReason;
};

} // namespace tengen::app
