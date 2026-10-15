#include "NetworkPresenter.hpp"

#include "gui/gameWidget.hpp"

#include <QMessageBox>
#include <QMetaObject>

namespace tengen {

static QString endText(const network::DisconnectReason reason) {
	switch (reason) {
	case network::DisconnectReason::ClosedLocally:
		return {};
	case network::DisconnectReason::ConnectFailed:
		return NetworkPresenter::tr("Could not reach the host.");
	case network::DisconnectReason::ConnectionLost:
		return NetworkPresenter::tr("The connection to the host was lost.");
	case network::DisconnectReason::ProtocolMismatch:
		return NetworkPresenter::tr("The host runs a different version of Tengen.");
	case network::DisconnectReason::ProtocolError:
		return NetworkPresenter::tr("The host sent something unexpected.");
	case network::DisconnectReason::Error:
		return NetworkPresenter::tr("A message from the host could not be handled.");
	}
	return {};
}

NetworkPresenter::NetworkPresenter(app::INetworkSession& session, gui::GameWidget& gameWidget)
    : m_session(session), m_gameWidget(gameWidget) {
	m_session.subscribe(this, app::AS_ConnectionChange);
}

NetworkPresenter::~NetworkPresenter() {
	m_session.unsubscribe(this);
}

void NetworkPresenter::onAppEvent(const app::AppSignalMask mask) {
	if (mask & app::AS_ConnectionChange) {
		QMetaObject::invokeMethod(this, [this, state = m_session.connection()] { showConnection(state); }, Qt::QueuedConnection);
	}
}

void NetworkPresenter::showConnection(const app::ConnectionState& state) {
	switch (state.phase) {
	case app::ConnectionPhase::Connecting:
		return;
	case app::ConnectionPhase::Connected:
		if (const auto player = state.localPlayer) {
			m_gameWidget.setPlayers(tr("You"), tr("Opponent"), *player);
		} else {
			m_gameWidget.setPlayers(tr("White"), tr("Black"), Player::White); // We only watch: name both sides by colour.
		}
		return;
	case app::ConnectionPhase::Idle:
		if (const auto reason = state.endReason; reason && *reason != network::DisconnectReason::ClosedLocally) {
			QMessageBox::warning(m_gameWidget.window(), tr("Disconnected"), endText(*reason));
		}
		return;
	}
}

} // namespace tengen
