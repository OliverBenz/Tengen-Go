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

NetworkPresenter::NetworkPresenter(app::INetworkSession& session, gui::GameWidget& gameWidget) : m_session(session), m_gameWidget(gameWidget) {
	m_session.subscribe(this, app::AS_ConnectionChange);
}

NetworkPresenter::~NetworkPresenter() {
	m_session.unsubscribe(this);
}

void NetworkPresenter::onAppEvent(const app::AppSignal signal) {
	if (signal != app::AS_ConnectionChange) {
		return;
	}
	// A copy per signal, so each ended connection shows one dialog, however late the GUI thread gets to it.
	QMetaObject::invokeMethod(this, [this, info = m_session.networkInfo()] { showConnection(info); }, Qt::QueuedConnection);
}

void NetworkPresenter::showConnection(const app::SessionNetworkInfo& info) {
	switch (info.getPhase()) {
	case app::ConnectionPhase::Connecting:
		return;
	case app::ConnectionPhase::Connected:
		if (const auto player = info.getLocalPlayer()) {
			m_gameWidget.setPlayers(tr("You"), tr("Opponent"), *player);
		} else {
			m_gameWidget.setPlayers(tr("White"), tr("Black"), Player::White); // We only watch: name both sides by colour.
		}
		return;
	case app::ConnectionPhase::Idle:
		if (const auto reason = info.getEndReason(); reason && *reason != network::DisconnectReason::ClosedLocally) {
			QMessageBox::warning(m_gameWidget.window(), tr("Disconnected"), endText(*reason));
		}
		return;
	}
}

} // namespace tengen
